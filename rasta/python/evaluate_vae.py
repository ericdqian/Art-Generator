from keras.models import load_model
from keras import backend as K
from keras.layers import Dense
from keras.models import Model
from models.alexnet import decaf
from keras import metrics
import numpy as np
import os
from os.path import join
import argparse
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import json
from datetime import datetime
from keras.preprocessing.image import load_img,img_to_array
from utils.utils import imagenet_preprocess_input,get_dico,wp_preprocess_input,invert_dico
from keras import activations

DEFAULT_MODEL_PATH='../savings/resnet_small/final_model.h5'
DEFAULT_BAGGING=True
DEFAULT_PREPROCESSING=None

def main():
    PATH = os.path.dirname(__file__)
    RESULT_FILE_PATH  = join(PATH,'../models/results.csv')
    K.set_image_data_format('channels_last')

    parser = argparse.ArgumentParser(description='Description')

    parser.add_argument('--isdecaf', action="store_true", dest='isdecaf',
                        help='if the model is a decaf6 type')
    parser.add_argument('-k', action="store", default=1, type=int, dest='k', help='top-k number')
    parser.add_argument('--data_path', action="store",
                        default=join(PATH, '../../data/vae_generated/MinimalismToBaroque'), dest='data_path',
                        help='Path of the data (image or train folder)')
    parser.add_argument('--model_path', action="store", dest='model_path', default=DEFAULT_MODEL_PATH,
                        help='Path of the h5 model file')
    parser.add_argument('-j', action="store_true", dest='json', help='Output prediction as json')
    parser.add_argument('-s', action="store_true", dest='save', help='Save accuracy in results file')
    parser.add_argument('-b', action="store_true", dest='b', default=DEFAULT_BAGGING, help='Sets bagging')
    parser.add_argument('-p', action="store", dest='preprocessing', default=DEFAULT_PREPROCESSING,
                        help='Type of preprocessing : [imagenet|wp]')
    parser.add_argument('-v', '--verbose', action="store_true", dest='verbose',
                        help='whether to print results to terminal')

    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    isdecaf = args.isdecaf # False
    k = args.k
    print(k)

    model = init(model_path, isdecaf)
    record_preds(model, data_path, isdecaf=isdecaf, top_k=k, bagging=args.b, preprocessing=args.preprocessing, use_json=args.json)
    

def record_preds(model, data_path, verbose=False, write=True, isdecaf=False, top_k=1, bagging=DEFAULT_BAGGING, preprocessing=DEFAULT_PREPROCESSING, use_json=False):
    with open(join(data_path, 'predictions.txt'), 'w+') as out:
        for img in os.listdir(data_path):
            full_path = join(data_path, img)
            if os.path.splitext(full_path)[-1].lower() in ['.png', '.jpg']:
                pred, pcts = get_pred(model, full_path, is_decaf6=isdecaf, top_k=top_k, bagging=bagging, preprocessing=preprocessing)
                if verbose:
                    print(pcts)
                
                if use_json:
                    result = { 'pred' : pred, 'top_k' : top_k }
                    print(json.dumps(result))
                else:
                    if verbose:
                        print(full_path)
                        print("Top-{} prediction : {}\n".format(top_k, pred))
                
                if write or not verbose:
                    out.write(str(pcts)+'\n')
                    out.write(full_path+'\n')
                    out.write("Top-{} prediction : {}\n\n".format(top_k, pred))

def get_pred(model, image_path, is_decaf6=False, top_k=1,bagging=DEFAULT_BAGGING,preprocessing=DEFAULT_PREPROCESSING):
    target_size = (224, 224)
    if is_decaf6:
        target_size = (227, 227)
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    if bagging:
        pred = _bagging_predict(x, model,preprocessing=preprocessing)
    else:
        x = _preprocess_input(x,preprocessing=preprocessing)
        pred = model.predict(x[np.newaxis, ...])
    dico = get_dico()
    inv_dico = invert_dico(dico)
    args_sorted = np.argsort(pred)[0][::-1]
    preds = [inv_dico.get(a) for a in args_sorted]
    pcts = [pred[0][a] for a in args_sorted]
    return preds,pcts

def _bagging_predict(x,model,preprocessing=None):
    x_flip = np.copy(x)
    x_flip = np.fliplr(x_flip)
    x = _preprocess_input(x,preprocessing)
    x_flip = _preprocess_input(x_flip,preprocessing)
    pred = model.predict(x[np.newaxis,...])
    pred_flip = model.predict(x_flip[np.newaxis,...])
    avg = np.mean(np.array([pred,pred_flip]), axis=0 )
    return avg

def _preprocess_input(x,preprocessing=None):
    if preprocessing == 'imagenet':
        return imagenet_preprocess_input(x)
    elif preprocessing == 'wp':
        return wp_preprocess_input(x)
    else:
        return x

def init(model_path, is_decaf6=False):
    if is_decaf6:
        K.set_image_data_format('channels_first')
        base_model = decaf()
        predictions = Dense(25, activation='softmax')(base_model.output)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path, by_name=True)
    else:
        model = load_model(model_path)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.top_k_categorical_accuracy])    
    return model

def draw_pred(model, data_path, folder_name):
    style_pcts = {}
    for img in os.listdir(data_path):
        full_path = join(data_path, img)
        if os.path.splitext(full_path)[-1].lower() in ['.png', '.jpg']:
            pred, pcts = get_pred(model, full_path)
            for style, pct in zip(pred, pcts):
                style_pcts.setdefault(style, []).append(pct)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ratios = ['{0:.2f}'.format(100*i/16) for i in range(16)]
    colors = {"Baroque": 'r-', "Cubism": 'b-', "Minimalism": 'g-'}
    for style, style_pcts in style_pcts.items():
        ax.plot(ratios, style_pcts, colors.get(style, '0-'), label=style)
    ax.set_xlabel('Ratio of {}'.format(folder_name))
    ax.set_ylabel('Prediction percentages')
    xticks = ax.get_xticks()[::2]
    ax.set_xticks(xticks)
    ax.set_xticklabels([0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100])
    # ax.set_title('Prediction percentages for images generated merging styles {}'.format(folder_name))
    fig.legend(title='Art style', loc='upper right')
    plt.tight_layout()
    # plt.show()
    fig.savefig(f'../../assets/vae_generated/{folder_name}.png', dpi = 300)
    print(folder_name)

if __name__ == '__main__':
    # main()

    # Calculates for all folders of images in a folder / Draws image for all
    PATH = os.path.dirname(__file__)
    RESULT_FILE_PATH  = join(PATH,'../models/results.csv')
    K.set_image_data_format('channels_last')

    model_path = DEFAULT_MODEL_PATH
    data_path = join(PATH, '../../data/vae_generated/')
    isdecaf = False
    k = 1

    model = init(model_path, isdecaf)

    for folder in os.listdir(data_path):
        folder_path = join(data_path, folder)
        if os.path.isdir(folder_path):
            # record_preds(model, folder_path, isdecaf=isdecaf, top_k=k, bagging=DEFAULT_BAGGING, preprocessing=DEFAULT_PREPROCESSING)
            draw_pred(model, folder_path, folder)
