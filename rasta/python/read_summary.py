import pickle
import argparse
import os
import matplotlib.pyplot as plt

SAVING_DIR = '../savings'

def graph_num_layers(measure='val_loss'):
    histories = []
    folder_prefix = 'resnet_rmsprop_n-trainable-'
    for num in [0, 20, 30]:
        folder_name = folder_prefix+str(num)
        pickle_path = os.path.join(SAVING_DIR, folder_name, 'history.pck')

        with open(pickle_path, 'rb') as pkl:
            history = pickle.load(pkl)
            histories.append(history)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    epochs = range(1, 11)
    ax.plot(epochs, histories[0][measure], 'r-', label='0')
    ax.plot(epochs, histories[1][measure], 'b-', label='20')
    ax.plot(epochs, histories[2][measure], 'g-', label='30')
    # ax.plot(epochs, histories[3][measure], 'm-', label='40')
    ax.set_xlabel('Epoch')

    if measure == 'val_loss':
        ylabel = 'Validation loss'
        ax.invert_yaxis()
    elif measure == 'val_acc':
        ylabel = 'Validation accuracy'
    elif measure == 'loss':
        ylabel = 'Training loss'
        ax.invert_yaxis()
    elif measure == 'acc':
        ylabel = 'Training accuracy'
    else:
        ylabel = measure

    ax.set_ylabel(ylabel)
    # ax.set_title('Validation loss over epochs')
    fig.legend(title='Number of extra retrainable layers', loc='upper right')
    plt.tight_layout()
    # plt.show()
    fig.savefig(f'../../assets/retrain_layers_{measure}.png', dpi = 300)

OPT_FOLDERS = ['resnet_rmsprop_n-trainable-30', 'resnet_sgd_mom_n-trainable-30', 'resnet_amsgrad_n-trainable-30']

def graph_opts(opt_folders=OPT_FOLDERS, measure='val_loss'):
    histories = []
    for folder_name in opt_folders:
        pickle_path = os.path.join(SAVING_DIR, folder_name, 'history.pck')

        with open(pickle_path, 'rb') as pkl:
            history = pickle.load(pkl)
            histories.append(history)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    epochs = range(1, 11)
    ax.plot(epochs, histories[0][measure], 'r-', label='RMSprop')
    ax.plot(epochs, histories[1][measure], 'b-', label='SGD + momentum')
    ax.plot(epochs, histories[2][measure], 'g-', label='AMSGrad')
    # ax.plot(epochs, histories[3][measure], 'm-', label='40')
    ax.set_xlabel('Epoch')

    if measure == 'val_loss':
        ylabel = 'Validation loss'
        ax.invert_yaxis()
    elif measure == 'val_acc':
        ylabel = 'Validation accuracy'
    elif measure == 'loss':
        ylabel = 'Training loss'
        ax.invert_yaxis()
    elif measure == 'acc':
        ylabel = 'Training accuracy'
    else:
        ylabel = measure

    ax.set_ylabel(ylabel)
    # ax.set_title('Validation loss over epochs')
    fig.legend(title='Optimizer', loc='upper right')
    plt.tight_layout()
    # plt.show()
    fig.savefig(f'../../assets/optimizers_{measure}.png', dpi = 300)

if __name__ == '__main__':
    # graph_num_layers(measure='val_loss')
    # graph_num_layers(measure='val_acc')
    # graph_num_layers(measure='loss')
    # graph_num_layers(measure='acc')

    graph_opts()
    graph_opts(measure='val_acc')

    # parser = argparse.ArgumentParser(description='Description')
    # parser.add_argument('-f', action="store", default='resnet_rmsprop_n-trainable-30/', dest='folder', 
    #     help='Folder under savings/')
    # args = parser.parse_args()
    
    # folder = args.folder
    # pickle_path = os.path.join(SAVING_DIR, folder, 'history.pck')
    # txt_path = os.path.join(SAVING_DIR, folder, 'history.txt')
    
    # with open(pickle_path, 'rb') as pkl, open(txt_path, 'w+') as txt:
    #     history = pickle.load(pkl)
    #     print(history)
    #     # txt.write(history)
