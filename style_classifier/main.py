import os
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

from data import WikiArtDataLoader, get_classes
from models import get_model

def train_model(model, dataloader, criterion, optimizer, scheduler, use_gpu=False, num_epochs=10, 
                    log_dir="runs/logs/cnn", log_filename="default"
                ):
    """ Trains model

    Args:
        model (nn.Module): model to train on
        dataloader: dataloader, with at least keys 'train' and 'valid'
        criterion: loss function, e.g. nn.CrossEntropyLoss()
        optimizer: optimizer function
        scheduler: learning rate scheduler
        num_epochs (int): number of epochs (passes thru data set) to perform
        use_gpu (bool): whether a GPU is being used or not
        log_dir (str): directory for logs
        log_filename (str): file name for log for this run
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_1 = 0.0

    best_model_wts_3 = copy.deepcopy(model.state_dict())
    best_acc_3 = 0.0

    best_model_wts_5 = copy.deepcopy(model.state_dict())
    best_acc_5 = 0.0

    create_dir(log_dir)
    log_file = os.path.join(log_dir, log_filename+'.txt')
    log = open(log_file, 'w+')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        log.write('Epoch {}/{}\n'.format(epoch + 1, num_epochs))
        log.write('-' * 10 + '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            print("start phase", phase)
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_1 = 0
            running_corrects_3 = 0
            running_corrects_5 = 0

            # # test data iter
            # iter_dl = iter(dataloader[phase])
            # test_data = []
            # for _ in range(10):
            #     test_data.append(next(iter_dl))
            # for inputs, labels in test_data:

            # Iterate over data
            for inputs, labels in dataloader[phase]:
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    acc1, acc3, acc5 = accuracy(outputs, labels, topk=(1,3, 5))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects_1 += acc1.data[0]
                running_corrects_3 += acc3.data[0]
                running_corrects_5 += acc5.data[0]

            epoch_loss = running_loss / len(dataloader[phase])
            epoch_acc_1 = running_corrects_1.double() / len(dataloader[phase])
            epoch_acc_3 = running_corrects_3.double() / len(dataloader[phase])
            epoch_acc_5 = running_corrects_5.double() / len(dataloader[phase])

            print('{} Loss: {:.4f} Acc@1: {:.4f} Acc@3: {:.4f} Acc@5: {:.4f}'.format(
                phase, epoch_loss, epoch_acc_1, epoch_acc_3, epoch_acc_5))

            log.write('{} Loss: {:.4f} Acc@1: {:.4f} Acc@3: {:.4f} Acc@5: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc_1, epoch_acc_3, epoch_acc_5))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=log_filename)

            # deep copy the model
            if phase == 'val':
                if epoch_acc_1 > best_acc_1:
                    best_acc_1 = epoch_acc_1
                    model_acc_3 = epoch_acc_3
                    model_acc_5 = epoch_acc_5
                    best_model_wts_1 = copy.deepcopy(model.state_dict())
                if epoch_acc_3 > best_acc_3:
                    best_acc_3 = epoch_acc_3
                    best_model_wts_3 = copy.deepcopy(model.state_dict())
                if epoch_acc_5 > best_acc_5:
                    best_acc_5 = epoch_acc_5
                    best_model_wts_5 = copy.deepcopy(model.state_dict())

        print()
        log.write('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc top-1: {:4f} top-3: {:4f} top-5 {:4f}'.format(best_acc_1, model_acc_3, model_acc_5))
    print('Best val Acc top-3: {:4f}'.format(best_acc_3))
    print('Best val Acc top-5: {:4f}'.format(best_acc_5))
    
    log.write(
        'Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)
    )
    log.write('Best val Acc top-1: {:4f} top-3: {:4f} top-5 {:4f}\n'.format(best_acc_1, model_acc_3, model_acc_5))
    log.write('Best val Acc top-3: {:4f}\n'.format(best_acc_3))
    log.write('Best val Acc top-5: {:4f}\n'.format(best_acc_5))

    log.close()

    # load best model weights
    model.load_state_dict(best_model_wts_1)
    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    
    Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best=False, checkpoint_dir='runs/checkpoints/', filename=''):
    '''
    a function to save checkpoint of the training
    :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()}
    :param is_best: boolean to save the checkpoint aside if it has the best score so far
    :param filename: the name of the saved file
    '''
    # create_dir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir + 'checkpoint_{}.pth.tar'.format(filename)))
    # if is_best:
    #     shutil.copyfile(self.args.checkpoint_dir + filename,
    #                     self.args.checkpoint_dir + 'model_best.pth.tar')

def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def run(perc=0.0, lr=0.01, where="end"):
    perc = 0.0
    lr = 0.01
    log_filename = 'perc-{}_lr-{}'.format(perc, lr)

    data_path = 'data/wikiart'

    use_gpu = False
    num_workers = 4
    pin_memory = False
    if torch.cuda.is_available():
        use_gpu = True
        num_workers = 1
        pin_memory = True
        print('Using GPU')

    wikiart_loader = WikiArtDataLoader(data_path, 32, (0.8, 0.2), random_seed=42, num_workers=num_workers, pin_memory=pin_memory)

    model = get_model("resnet50", data_path, percentage_retrain=perc, where=where_retrain)

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = train_model(model, wikiart_loader, criterion, optimizer, exp_lr_scheduler, use_gpu=use_gpu, num_epochs=10, log_dir="runs/logs/cnn", log_filename=log_filename)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'retrain': 'last_layer',
        'optimizer': 'sgd',
        'init_lr': lr,
        # 'momentum': 0.9,
        'steplr_size': 5,
        'steplr_gamma': 0.1,
        'perc': perc,
        'where': where_retrain,
    }, 'runs/models/resnet50_{}.pth.tar'.format(log_filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--perc', action="store", dest='perc', default=0.0,
                        help='Percentage of layers from output to retrain')
    parser.add_argument('--lr', action="store", dest='lr', default=0.01,
                        help='Learning rate')
    parser.add_argument('--where_retrain', action="store", dest='where_retrain', default="end",
                        help='Where in model the layers are retrained')

    args = parser.parse_args()

    perc = args.perc
    lr = args.lr
    where_retrain = args.where_retrain

    run(perc=perc, lr=lr, where=where_retrain)
