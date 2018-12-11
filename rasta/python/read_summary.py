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

def draw():
    counts = get_counts()
    styles = []
    style_counts = []
    for style, cnt in counts.items():
        styles.append(style)
        style_counts.append(cnt)

    # plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    y_pos = range(len(styles))

    ax.barh(y_pos, style_counts, label='wikiart', align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(styles)
    # fig.legend(loc='upper right')
    ax.invert_yaxis()  # labels read top-to-bottom
    # label bars with counts
    for i, v in enumerate(style_counts):
        ax.text(v + 3, i + .25, str(v), color='black')
    ax.set_xlabel('Number of images')
    ax.set_title('Distribution of images to art styles')
    plt.xlim((0,15000))
    plt.tight_layout()
    # plt.show()
    fig.savefig('assets/img_distrib.png', dpi = 300)  

if __name__ == '__main__':
    # graph_num_layers(measure='val_loss')
    # graph_num_layers(measure='val_acc')
    # graph_num_layers(measure='loss')
    # graph_num_layers(measure='acc')
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-f', action="store", default='resnet_rmsprop_n-trainable-30/', dest='folder', 
        help='Folder under savings/')
    args = parser.parse_args()
    
    folder = args.folder
    pickle_path = os.path.join(SAVING_DIR, folder, 'history.pck')
    txt_path = os.path.join(SAVING_DIR, folder, 'history.txt')
    
    with open(pickle_path, 'rb') as pkl, open(txt_path, 'w+') as txt:
        history = pickle.load(pkl)
        print(history)
        # txt.write(history)
