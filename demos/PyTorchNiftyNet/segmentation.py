import time, argparse
import numpy as np

from libs import dataset_niftynet as dset_utils
from libs import loss as loss_utils
from libs import model as cnn_utils

from niftynet.engine.sampler_balanced_v2 import BalancedSampler
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.mean_variance_normalisation import MeanVarNormalisationLayer
from niftynet.engine.signal import TRAIN, VALID, INFER
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.evaluation.pairwise_measures import PairwiseMeasures


import torch
from torch.utils.data import DataLoader
import torch.optim as optim


def get_reader(data_param,image_sets_partitioner,phase):

    # Using Nifty Reader
    if phase == 'training':
        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(TRAIN))

    elif phase == 'validation':
        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(VALID))

    elif phase == 'inference':

        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(INFER))
    else:
        raise Exception('Invalid phase choice: {}'.format({'phase':['train','validation','inference']}))


    # Adding preprocessing layers
    mean_variance_norm_layer = MeanVarNormalisationLayer(image_name='image')
    image_reader.add_preprocessing_layers([mean_variance_norm_layer])

    return image_reader

def get_sampler(image_reader,patch_size, phase):


    if phase == 'training' or phase == 'validation':

        sampler = BalancedSampler(image_reader,
                                 window_sizes=patch_size,
                                 windows_per_image=4)
    elif phase == 'inference':

        sampler = GridSampler(image_reader,
                              window_sizes=patch_size,
                              window_border=(8,8,8),
                              batch_size=1)
    else:
        raise Exception('Invalid phase choice: {}'.format({'phase': ['train', 'validation', 'inference']}))


    return sampler


def train(dsets,model,criterion,optimizer,num_epochs,device,cp_path,batch_size):

    since = time.time()

    dataloaders = {x: DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                  for x in ['training', 'validation']}

    model = model.to(device)


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            epoch_samples = 0

            # Iterate over data
            for iteration, (inputs, labels) in enumerate(dataloaders[phase], 1):

                nbatches, wsize, x, y, z, nchannels = inputs.size()

                inputs = inputs.view(nbatches * wsize, nchannels, x,y,z)
                labels = labels.view(nbatches * wsize, nchannels, x,y,z)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    pred = (outputs > 0.5)

                    los = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        los.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                running_loss += los.item()*inputs.size(0)
                running_corrects += PairwiseMeasures(pred.cpu().numpy(),
                                                     labels.cpu().numpy()).dice_score()*inputs.size(0)


            epoch_loss = running_loss / (epoch_samples)

            epoch_acc = running_corrects / (epoch_samples)

            print('{} Loss: {:.4f} Dice: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if epoch == 0:
                best_loss = epoch_loss
                torch.save(model.state_dict(), cp_path.format(epoch + 1))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), cp_path)
                print('Checkpoint {} saved !'.format(epoch + 1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def inference(sampler,model,device,pred_path,cp_path):

    output = GridSamplesAggregator(image_reader=sampler.reader,
                                           output_path=pred_path)

    model.load_state_dict(torch.load(cp_path))
    model.to(device)
    model.eval()

    for batch_output in sampler():
        window =  batch_output['image'][...,0,:] # [...,0,:] eliminates time coordinate from NiftyNet Volume

        nb, x, y, z, nc = window.shape
        window = window.reshape(nb, nc, x, y, z)
        window = torch.Tensor(window).to(device)

        with torch.no_grad():
            outputs = model(window)
            outputs = (outputs > 0.5)

        output.decode_batch(outputs.cpu().numpy().reshape(nb, x, y, z, nc).astype(np.uint8),
                                    batch_output['image_location'])



def main():

    opt = parsing_data()

    print("[INFO]Reading data")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    else:
        raise Exception("[INFO] No GPU found or Wrong gpu id, please run without --cuda")



    # Dictionary with data parameters for NiftyNet Reader
    data_param = {'image': {'path_to_search': opt.image_path,
                            'filename_contains': 'CC'},
                  'label': {'path_to_search': opt.label_path,
                            'filename_contains': 'CC'},
                  'sampler': {'path_to_search': opt.label_path, # binary weights for sampling
                            'filename_contains': 'CC'}}

    # Partitioning dataset using NiftyNet
    image_sets_partitioner = ImageSetsPartitioner().initialise(data_param=data_param,
                                                               data_split_file=opt.data_split_file,
                                                               new_partition=False,
                                                               ratios=opt.ratios)

    readers = {x: get_reader(data_param,image_sets_partitioner,x)
               for x in ['training', 'validation','inference']}
    samplers = {x: get_sampler(readers[x],opt.patch_size,x)
               for x in ['training', 'validation','inference']}

    # Training stage only
    dsets = {x: dset_utils.DatasetNiftySampler(sampler=samplers[x])
             for x in ['training', 'validation']}


    print("[INFO] Building model")
    model = cnn_utils.Modified3DUNet(opt.in_channels,opt. n_classes)
    criterion =  loss_utils.SoftDiceLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)

    print("[INFO] Training")
    train(dsets,model,criterion,optimizer,opt.num_epochs,device,opt.cp_path,opt.batch_size)


    print("[INFO] Inference")
    inference(samplers['inference'],model,device,opt.pred_path,opt.cp_path)


def parsing_data():

    parser = argparse.ArgumentParser(description='3D Segmentation Using PyTorch and NiftyNet')
    parser.add_argument('-data_split_file', default='train_val_infer_split.csv',
                        type=str, help='output csv filename')
    parser.add_argument('-patch_size', default=(64, 64, 64),
                        type=tuple, help='patch size')
    parser.add_argument('-in_channels', default=1,
                        type=int, help='# of data channels')
    parser.add_argument('-n_classes', default=1,
                        type=int, help='# of output classes')
    parser.add_argument('-num_epochs', default=6,
                        type=int, help='# of epochs')
    parser.add_argument('-lr', default=1e-4,
                        type=float, help='learning rate')
    parser.add_argument('-ratios', default=[0.1, 0.1],
                        type=list, help='ratio for validation and inference sets')
    parser.add_argument('-batch_size', default=4,
                        type=int, help='batch size')
    parser.add_argument('-cp_path', default='./CP.pth',
                        type=str, help='checkpoint output filename')
    parser.add_argument('-image_path', default='./CC359_NEW/data',
                        type=str, help='image path')
    parser.add_argument('-label_path', default='./CC359_NEW/STAPLE-binary',
                        type=str, help='label path')
    parser.add_argument('-pred_path', default='./pred',
                        type=str, help='output path for inferences')

    opt = parser.parse_args()

    return  opt


if __name__ == '__main__':
    main()
