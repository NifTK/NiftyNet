#import tensorflow as tf
#from base import Layer
#
#
#class AddXLayer(Layer):
#    def __init__(self, X, name='add_X_layer'):
#        super(AddXLayer, self).__init__(name=name)
#        self.X = X
#        self.myvar = None
#
#    def layer_op(self, input_tensor):
#        self.myvar = tf.constant(name='X', value=self.X)
#        return input_tensor + self.myvar
#
#
#class AddAddLayer(Layer):
#    def __init__(self, X, Y, name='add_add_layer'):
#        super(AddAddLayer, self).__init__(name=name)
#        self.X = X
#        self.Y = Y
#
#    def layer_op(self, input_tensor):
#        # create component layers
#        first_add_layer = AddXLayer(self.X, name='first_add')
#        second_add_layer = AddXLayer(self.Y, name='second_add')
#
#        # initialise component layers
#        first_out = first_add_layer(input_tensor)
#        second_out = second_add_layer(first_out)
#        return second_out
#
#
#class BasicLayer(Layer):
#    def __init__(self, X, Y, basic_op, name='basic_layer_'):
#        new_name = name + basic_op.__name__
#        super(BasicLayer, self).__init__(name=new_name)
#        self.X = X
#        self.Y = Y
#        self.basic_op = basic_op
#
#    def layer_op(self, input_x, input_y):
#        # create component layers
#        output = self.basic_op(self.X * input_x, self.Y * input_y)
#        return output
#
#
#class VarLayer(Layer):
#    def __init__(self, initializer_x=None, name='variable_layer'):
#        super(VarLayer, self).__init__(name=name)
#        if initializer_x:
#            self.init_x = initializer_x
#        else:
#            self.init_x = tf.random_uniform_initializer(0.0)
#
#    def layer_op(self, input_x):
#        # create component layers
#        input_shape = input_x.get_shape()
#        myvar1 = tf.get_variable(
#            'my_var_1', shape=input_shape, initializer=self.init_x)
#        output = input_x * myvar1
#        return output
#
#
#class TwoVarLayer(Layer):
#    def __init__(self, initializer_x=None, name='two_variable_layer'):
#        super(TwoVarLayer, self).__init__(name=name)
#        if initializer_x:
#            self.init_x = initializer_x
#        else:
#            self.init_x = tf.random_uniform_initializer(0.0)
#
#    def layer_op(self, input_x):
#        # create component layers
#        first_var_op = VarLayer(self.init_x, name='first_var_layer')
#        second_var_op = VarLayer(self.init_x, name='second_var_layer')
#        first_out = first_var_op(input_x)
#        second_out = second_var_op(first_out)
#        return second_out
#
#
#with tf.variable_scope('test_1') as scope:
#    my_add = AddXLayer(X=12.0)
#    print my_add.to_string()
#
#    out_1 = my_add(tf.constant([1.0, 1.0]))
#    out_2 = my_add(tf.constant([2.0, 2.0]))
#    print my_add.to_string()
#
#    my_add_add = AddAddLayer(X=13.0, Y=100.0)
#    out_3 = my_add_add(tf.constant([3.0, 3.0]))
#    print my_add_add.to_string()
#
#    my_basic_op = BasicLayer(2.0, 3.0, tf.add)
#    out_4 = my_basic_op(tf.constant(1.0), tf.constant(1.0))
#    print my_basic_op.to_string()
#
#    my_var_op = VarLayer()
#    out_5 = my_var_op(tf.constant([2.0, 1.0]))
#    print my_var_op.to_string()
#
#    my_two_var_op = TwoVarLayer()
#    out_6 = my_two_var_op(tf.constant([2.0, 1.0]))
#    print my_two_var_op.to_string()
#
#    init_op = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init_op)
# print sess.run(out_1)
# print sess.run(out_2)
# print sess.run(out_3)
# print sess.run(out_4)
# print sess.run(out_5)
# print sess.run(out_6)
