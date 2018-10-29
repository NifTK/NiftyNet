import time
import numpy as np

from libs import dataset_niftynet as dset_utils
from libs import loss as loss_utils
from libs import model as cnn_utils

from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.mean_variance_normalisation import MeanVarNormalisationLayer
from niftynet.layer.rand_rotation import RandomRotationLayer as Rotate
from niftynet.engine.signal import TRAIN, VALID, INFER
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import BinaryAccuracy, Loss



def get_sampler(data_param,image_sets_partitioner,phase,patch_size):


    # Using Nifty Reader and Sampler to create a dataset for PyTorch DataLoader
    if phase == 'train':
        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(TRAIN))

    elif phase == 'validation':
        image_reader = ImageReader().initialise(data_param,
                                                file_list=image_sets_partitioner.get_file_list(VALID))
    else:
        raise Exception('Invalid phase choice: {}'.format({'phase':['train','validation']}))


    mean_variance_norm_layer = MeanVarNormalisationLayer(image_name='image')
    rotation_layer = Rotate()
    rotation_layer.init_uniform_angle([-10.0, 10.0])

    image_reader.add_preprocessing_layers([mean_variance_norm_layer]) # preprocessing
    image_reader.add_preprocessing_layers([rotation_layer]) # augmentation


    sampler = UniformSampler(image_reader,
                             window_sizes=(patch_size, patch_size, patch_size),
                             windows_per_image=1)

    return sampler


def create_summary_writer(model, data_loader):
    writer = SummaryWriter()
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def run(dataloaders, model,criterion,optimizer,num_epochs,device):
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    writer = create_summary_writer(model, train_loader)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'lossDice': Loss(criterion),
                                                     'acc': BinaryAccuracy()},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % 10 == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['acc']
        avg_nll = metrics['lossDice']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['acc']
        avg_nll = metrics['lossDice']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=num_epochs)
    writer.close()




def inference(data_param, image_sets_partitioner, patch_size, window_border, model, device):

    image_reader_test = ImageReader(names='image').initialise(data_param,
                                                              file_list=image_sets_partitioner.get_file_list(INFER))
    sampler_test = GridSampler(image_reader_test,
                               window_sizes=(patch_size, patch_size, patch_size),
                               window_border=window_border,
                               batch_size = 1)

    output_decoder = GridSamplesAggregator(image_reader=sampler_test.reader,
                                           output_path='/home/oeslle/Documents/pred')

    model.load_state_dict(torch.load('./CP2.pth'))
    model.to(device)
    model.eval()

    for batch_output in sampler_test():
        window =  batch_output['image'][...,0,:] # [...,0,:] eliminates time coordinate from NiftyNet Volume

        nb, x, y, z, nc = window.shape
        window = window.reshape(nb, nc, x, y, z)
        window = torch.Tensor(window).to(device)

        with torch.no_grad():
            outputs = model(window)[1]
            outputs = F.sigmoid(outputs)
            outputs = (outputs > 0.5)

        output_decoder.decode_batch(outputs.cpu().numpy().reshape(nb, x, y, z, nc).astype(np.uint8),
                                    batch_output['image_location'])



def main():

    print("[INFO]Reading data")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    else:
        raise Exception("[INFO] No GPU found or Wrong gpu id, please run without --cuda")

    # Dictionary with data parameters for NiftyNet Reader
    data_param = {'image': {'path_to_search': '/home/oeslle/Documents/Datasets/CC359_NEW/data',
                            'filename_contains': 'CC'},
                  'label': {'path_to_search': '/home/oeslle/Documents/Datasets/CC359_NEW/label',
                            'filename_contains': 'CC'}}

    # NiftyNet parameters
    data_split_file = 'train_val_infer_split.csv'
    patch_size = 32
    window_border = (8,8,8)

    print("[INFO] Building model")
    in_channels = 1
    n_classes = 1
    num_epochs = 2
    lr = 1e-6
    model = cnn_utils.Modified3DUNet(in_channels, n_classes)
    criterion =  loss_utils.SoftDiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)


    # Partitioning dataset using NiftyNet
    image_sets_partitioner = ImageSetsPartitioner().initialise(data_param=data_param,
                                                               data_split_file=data_split_file,
                                                               new_partition=False,
                                                               ratios=[0.1, 0.1])


    dsets = {'train':dset_utils.DatasetNiftySampler(sampler=get_sampler(data_param,
                                                                        image_sets_partitioner,
                                                                        'train',
                                                                        patch_size)),
             'val':dset_utils.DatasetNiftySampler(sampler=get_sampler(data_param,
                                                                      image_sets_partitioner,
                                                                      'validation',
                                                                      patch_size))}

    # Using PytTorch DataLoader
    dataloaders = {x: DataLoader(dsets[x], batch_size=4, shuffle=True, num_workers=3)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}


    print("[INFO] Training")
    run(dataloaders, model, criterion, optimizer, num_epochs, device)
    # train(dataloaders,dataset_sizes,model,criterion,optimizer,num_epochs,device,in_channels)

    # print("[INFO] Inference")
    # inference(data_param, image_sets_partitioner, patch_size, window_border, model, device)


if __name__ == '__main__':
    main()


