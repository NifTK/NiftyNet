from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import function
import numpy

"""
Re-implementation of [1] for volumetric image processing.


[1] Zheng, Shuai, et al. "Conditional random fields as recurrent neural networks." 
CVPR 2015.
"""

def permutohedral_prepare(position_vectors):
    batch_size = int(position_vectors.get_shape()[0])
    nCh=int(position_vectors.get_shape()[-1])
    nVoxels=int(position_vectors.get_shape().num_elements())//batch_size//nCh
    # reshaping batches and voxels into one dimension means we can use 1D gather and hashing easily
    position_vectors=tf.reshape(position_vectors,[-1,nCh])

    ## Generate position vectors in lattice space
    x=position_vectors/(numpy.sqrt(2./3.)*(nCh+1))
    
    # Embed in lattice space using black magic from the permutohedral paper
    alpha=lambda i:numpy.sqrt(float(i)/(float(i)+1.))
    Ex=[None]*(nCh+1)
    Ex[nCh]=-alpha(nCh)*x[:,nCh-1]
    for dit in range(nCh-1,0,-1):
        Ex[dit]=-alpha(dit)*x[:,dit-1]+x[:,dit]/alpha(dit+1)+Ex[dit+1]

    Ex[0]=x[:,0]/alpha(1)+Ex[1]
    Ex=tf.stack(Ex,1)
    ## Compute coordinates
    # Get closest remainder-0 point
    v=tf.to_int32(tf.round(Ex*(1./float(nCh+1))))
    rem0=v*(nCh+1)
    sumV=tf.reduce_sum(v,1,keep_dims=True)
    # Find the simplex we are in and store it in rank (where rank describes what position coorinate i has
    #in the sorted order of the features values)
    di=Ex-tf.to_float(rem0)
    _,index=tf.nn.top_k(di,nCh+1,sorted=True) 
    _,rank=tf.nn.top_k(-index,nCh+1,sorted=True) # This can be done more efficiently if necessary following the permutohedral paper

    # if the point doesn't lie on the plane (sum != 0) bring it back 
    rank=tf.to_int32(rank)+sumV;
    addMinusSub=tf.to_int32(rank<0)*(nCh+1)-tf.to_int32(rank>=nCh+1)*(nCh+1)
    rank=rank+addMinusSub
    rem0=rem0+addMinusSub

    # Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    v2=(Ex-tf.to_float(rem0))*(1./float(nCh+1))
    # the barycentric coordinates are v_sorted-v_sorted[...,[-1,1:-1]]+[1,0,0,...]
    # CRF2RNN uses the calculated ranks to get v2 sorted in O(nCh) time
    # We cheat here by using the easy to implement but slower method of sorting again in O(nCh log nCh)
    # we might get this even more efficient if we correct the original sorted data above
    v_sortedDesc,_=tf.nn.top_k(v2,nCh+1,sorted=True)
    v_sorted=tf.reverse(v_sortedDesc,[1])
    barycentric=v_sorted-tf.concat([v_sorted[:,-1:]-1.,v_sorted[:,0:nCh]],1)

    # Compute all vertices and their offset
    canonical = [[i]*(nCh+1-i)+[i-nCh-1]*i for i in range(nCh+1)]
    # WARNING: This hash function does not guarantee uniqueness of different position_vectors
    hashVector = tf.constant(numpy.power(int(numpy.floor(numpy.power(tf.int64.max,1./(nCh+2)))),[range(1,nCh+1)]),dtype=tf.int64)
    hash=lambda key: tf.reduce_sum(tf.to_int64(key)*hashVector,1)
    hashtable=tf.contrib.lookup.MutableDenseHashTable(tf.int64,tf.int64,default_value=tf.constant([-1]*nCh,dtype=tf.int64),empty_key=-1,initial_num_buckets=8,checkpoint=False)
    indextable=tf.contrib.lookup.MutableDenseHashTable(tf.int64,tf.int64,default_value=0,empty_key=-1,initial_num_buckets=8,checkpoint=False)

    numSimplexCorners=nCh+1
    keys=[None]*numSimplexCorners
    i64keys=[None]*numSimplexCorners
    insertOps=[]
    for scit in range(numSimplexCorners):
        keys[scit] = tf.gather(canonical[scit],rank[:,:-1])+rem0[:,:-1]
        i64keys[scit]=hash(keys[scit])
        insertOps.append(hashtable.insert(i64keys[scit],tf.to_int64(keys[scit])))

    with tf.control_dependencies(insertOps):
        fusedI64Keys,fusedKeys = hashtable.export()
        fusedKeys=tf.boolean_mask(fusedKeys,tf.not_equal(fusedI64Keys,-1))
        fusedI64Keys=tf.boolean_mask(fusedI64Keys,tf.not_equal(fusedI64Keys,-1))

    insertIndices = indextable.insert(fusedI64Keys,tf.expand_dims(tf.transpose(tf.range(1,tf.to_int64(tf.size(fusedI64Keys)+1),dtype=tf.int64)),1))                                
    blurNeighbours1=[None]*(nCh+1)
    blurNeighbours2=[None]*(nCh+1)
    indices=[None]*(nCh+1)
    with tf.control_dependencies([insertIndices]):
        for dit in range(nCh+1):
            offset=tf.constant([nCh if i==dit else -1 for i in range(nCh)],dtype=tf.int64)
            blurNeighbours1[dit]=indextable.lookup(hash(fusedKeys+offset))
            blurNeighbours2[dit]=indextable.lookup(hash(fusedKeys-offset))
            batch_index=tf.reshape(tf.meshgrid(tf.range(batch_size),tf.zeros([nVoxels],dtype=tf.int32))[0],[-1,1])
            indices[dit] = tf.stack([tf.to_int32(indextable.lookup(i64keys[dit])),batch_index[:,0]],1) # where in the splat variable each simplex vertex is
    return barycentric,blurNeighbours1,blurNeighbours2,indices
        
def permutohedral_compute(data_vectors,barycentric,blurNeighbours1,blurNeighbours2,indices,name,reverse):
    batch_size=tf.shape(data_vectors)[0]
    numSimplexCorners=int(barycentric.get_shape()[-1])
    nCh=numSimplexCorners-1
    nChData=tf.shape(data_vectors)[-1]
    data_vectors = tf.reshape(data_vectors,[-1,nChData])
    data_vectors = tf.concat([data_vectors,tf.ones_like(data_vectors[:,0:1])],1) # Convert to homogenous coordinates
    ## Splatting
    initialSplat=tf.zeros([tf.shape(blurNeighbours1[0])[0]+1,batch_size,nChData+1])
    with tf.variable_scope(name):
        # WARNING: we use local variables so the graph must initialize local variables with tf.local_variables_initializer()
        splat=tf.contrib.framework.local_variable(tf.ones([0,0]),validate_shape=False,name='splatbuffer')

    with tf.control_dependencies([splat.initialized_value()]):
        resetSplat=tf.assign(splat,initialSplat,validate_shape=False,name='assign')
    # This is needed to force tensorflow to update the cache
    with tf.control_dependencies([resetSplat]):
        uncachedSplat=splat.read_value()
    for scit in range(numSimplexCorners):
        data = data_vectors*barycentric[:,scit:scit+1]
        with tf.control_dependencies([uncachedSplat]):
            splat=tf.scatter_nd_add(splat,indices[scit],data)

    ## Blur
    with tf.control_dependencies([splat]):
        blurred=[splat]
        order = range(nCh,-1,-1) if reverse else range(nCh+1)
        for dit in order:
            with tf.control_dependencies([blurred[-1]]):
                b1=0.5*tf.gather(blurred[-1],blurNeighbours1[dit])
                b2=blurred[-1][1:,:,:]
                b3=0.5*tf.gather(blurred[-1],blurNeighbours2[dit])
                blurred.append(tf.concat([blurred[-1][0:1,:,:], b2+b1+b3],0))

    # Alpha is a magic scaling constant from CRFAsRNN code
    alpha = 1. / (1.+numpy.power(2., -nCh))
    normalized=blurred[-1][:,:,:-1]/blurred[-1][:,:,-1:]
    ## Slice
    sliced = tf.gather_nd(normalized,indices[0])*barycentric[:,0:1]*alpha
    for scit in range(1,numSimplexCorners):
        sliced = sliced+tf.gather_nd(normalized,indices[scit])*barycentric[:,scit:scit+1]*alpha

    return sliced

# Differentiation can be done using permutohedral lattice with gaussion filter order reversed
# To get this to work with automatic differentiation we use a hack attributed to Sergey Ioffe 
# mentioned here: http://stackoverflow.com/questions/36456436/how-can-i-define-only-the-gradient-for-a-tensorflow-subgraph/36480182

# Define custom py_func which takes also a grad op as argument: from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(numpy.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def gradientStub(data_vectors,barycentric,blurNeighbours1,blurNeighbours2,indices,name):
# This is a stub operator whose purpose is to allow us to overwrite the gradient. 
# The forward pass gives zeros and the backward pass gives the correct gradients for the permutohedral_compute function
    return py_func(lambda data_vectors,barycentric,blurNeighbours1,blurNeighbours2,indices: data_vectors*0,
                        [data_vectors,barycentric,blurNeighbours1,blurNeighbours2,indices],
                        [tf.float32],
                        name=name,
                        grad=lambda op,grad: [permutohedral_compute(grad,op.inputs[1],op.inputs[2],op.inputs[3],op.inputs[4],name,reverse=True)]+[tf.zeros_like(i) for i in op.inputs[1:]])


def permutohedral_gen(permutohedral, data_vectors,name):
    barycentric,blurNeighbours1,blurNeighbours2,indices=permutohedral
    return gradientStub(data_vectors,barycentric,blurNeighbours1,blurNeighbours2,indices,name)+ tf.stop_gradient(tf.reshape(permutohedral_compute(data_vectors,barycentric,blurNeighbours1,blurNeighbours2,indices,name,reverse=False),data_vectors.get_shape()))
    
    
def ftheta(U,H1,permutohedrals,mu,kernel_weights, aspect_ratio,name):
    nCh=U.get_shape().as_list()[-1]
    batch_size=int(U.get_shape()[0])
    # Message Passing
    data=tf.reshape(tf.nn.softmax(H1),[batch_size,-1,nCh])
    Q1=[None]*len(permutohedrals)
    with tf.device('/cpu:0'):
        for idx,permutohedral in enumerate(permutohedrals):
            Q1[idx] = tf.reshape(permutohedral_gen(permutohedral,data,name+str(idx)),U.get_shape())
    # Weighting Filter Outputs
    Q2=tf.add_n([Q1*w for Q1,w in zip(Q1,kernel_weights)])
    # Compatibility Transform
    Q3=tf.nn.conv3d(Q2,mu,strides=[1,1,1,1,1],padding='SAME')
    # Adding Unary Potentials
    Q4=U-Q3
    # Normalizing
    return Q4 # output logits, not the softmax

class CRFAsRNNLayer(TrainableLayer):
    """
    This class defines a layer implementing CRFAsRNN described in [1] using 
    a bilateral and a spatial kernel as in [2].
    Essentially, this layer smooths its input based on a distance in a feature
    space comprising spatial and feature dimensions.
    [1] Zheng, Shuai, et al. "Conditional random fields as recurrent neural networks." CVPR 2015.
    [2] https://arxiv.org/pdf/1210.5644.pdf
    """
    def __init__(self,alpha=5.,beta=5.,gamma=5.,T=5,aspect_ratio=[1.,1.,1.], name="crf_as_rnn"):
      """
      Parameters: 
      alpha:        bandwidth for spatial coordinates in bilateral kernel. 
                      Higher values cause more spatial blurring
      beta:         bandwidth for feature coordinates in bilateral kernel
                      Higher values cause more feature blurring
      gamma:        bandwidth for spatial coordinates in spatial kernel
                      Higher values cause more spatial blurring
      T: number of stacked layers in the RNN
      aspect_ratio: spacing of adjacent voxels (allows isotropic spatial smoothing when voxels are
                    not isotropic
      """
      super(CRFAsRNNLayer, self).__init__(name=name)
      self._T=T
      self._aspect_ratio=aspect_ratio
      self._alpha=alpha
      self._beta=beta
      self._gamma=gamma
      self._name=name
    def layer_op(self, I,U):
      """
      Parameters:
        I: feature maps defining the non-spatial dimensions within which smoothing is performed
           For example, to smooth U within regions of similar intensity this would be the
           image intensity 
        U: activation maps to smooth
      """
      batch_size=int(U.get_shape()[0])
      H1=[U]
      # Build permutohedral structures for smoothing
      coords=tf.tile(tf.expand_dims(tf.stack(tf.meshgrid(*[numpy.array(range(int(i)),dtype=numpy.float32)*a for i,a in zip(U.get_shape()[1:4],self._aspect_ratio)]),3),0),[batch_size,1,1,1,1])
      bilateralCoords =tf.reshape(tf.concat([coords/self._alpha,I/self._beta],4),[batch_size,-1,int(I.get_shape()[-1])+3])
      spatialCoords=tf.reshape(coords/self._gamma,[batch_size,-1,3])
      kernel_coords=[bilateralCoords,spatialCoords]
      permutohedrals = [permutohedral_prepare(coords) for coords in kernel_coords]

      nCh=U.get_shape()[-1]
      mu = tf.get_variable('Compatibility',initializer=tf.constant(numpy.reshape(numpy.eye(nCh),[1,1,1,nCh,nCh]),dtype=tf.float32));
      kernel_weights = [tf.get_variable("FilterWeights"+str(idx), shape=[1,1,1,1,nCh], initializer=tf.zeros_initializer()) for idx,k in enumerate(permutohedrals)]
    
      for t in range(self._T):
        H1.append(ftheta(U,H1[-1],permutohedrals,mu,kernel_weights, aspect_ratio=self._aspect_ratio,name=self._name+str(t)))
      return H1[-1]
    

    
    
