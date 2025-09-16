#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""

"""

import os
import pickle
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re

plt.ion()   # interactive mode


def save_variable(filename, var):
    print(filename)
    pickle_f = open(str(filename), 'wb')
    pickle.dump(var, pickle_f)
    pickle_f.close()
    return filename


def load_variable(filename):
    pickle_f = open(filename, 'rb')
    var = pickle.load(pickle_f)
    pickle_f.close()
    return var


train_data_transforms = transforms.Compose([
    transforms.Resize(size=(1000, 1000)),
    transforms.RandomRotation(30),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.6347, 0.4836, 0.6046], [0.2721, 0.2595, 0.2631])
])
data_transforms = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.7080, 0.5029, 0.6375], [0.1735, 0.2072, 0.1628])
])


def convert_label(_label, _mode, _class_num, _class_cate):
    if _mode:
        conv_label = []
        for _it in _label.split(','):
            for i in range(_class_num):
                if _it == str(list(_class_cate)[i]):
                    conv_label.append(1)
                else:
                    conv_label.append(0)
        return conv_label
    else:
        for i in range(_class_num):
            if str(_label) == list(_class_cate)[i]:
                conv_label = i
                return conv_label


class ImageDataset(Dataset):

    def __init__(self, dataframe=None, transform=None, class_cate=None, class_num = None, mode=None):
        self.dataframe = dataframe
        self.transform = transform
        self._work_mode = mode
        self._class_cate = class_cate
        self._class_number = class_num

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = io.imread(img_path)
        img = Image.fromarray(image)
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        _label = convert_label(label, self._work_mode, self._class_number, self._class_cate)

        sample = {"image": img, "label": _label}
        return sample


def load_data(_class_cate, _class_number, _work_mode):
    train_img_label_df = pd.read_csv("HEAL_Workspace/outputs/train_fold_0.csv")
    val_img_label_df = pd.read_csv("HEAL_Workspace/outputs/val_fold_0.csv")
    train_dataset = ImageDataset(dataframe=train_img_label_df, transform=train_data_transforms,
                                 class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
    val_dataset = ImageDataset(dataframe=val_img_label_df, transform=data_transforms,
                               class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
    # test_dataset = ImageDataset(dataframe=test_img_label_df, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)
    # test_loader = DataLoader(test_dataset, batch_size=8,shuffle=False, num_workers=16)
    return train_loader, val_loader

pattern = re.compile(r'\..*jpeg')
def get_test_sample(_patient, _label):
    print(_patient)
    _test_imgs_df = pd.DataFrame(columns=['Image_path', 'Label'])
    try:
        for _files in os.listdir(_patient):
            global pattern
            if _files.split('.')[-1] == 'jpeg' and not pattern.match(_files):
                _img_path = os.path.join(_patient, _files)
                _tmp = pd.DataFrame([[_img_path, _label]], columns=['Image_path', 'Label'])
                _test_imgs_df = pd.concat([_test_imgs_df, _tmp])
        return _test_imgs_df
    except Exception as e:
        print(e)
        return _test_imgs_df


def tensor2array(t, num):
    t = t.cpu()
    ar = [i.detach().numpy() for i in t]
    tmp = []
    for i in range(len(ar)):
        tmp = np.concatenate((tmp, ar[i]), axis=None)
    ar = tmp
    ar = ar.reshape((-1, num))
    return ar


def plot_roc_curve(pred_y, test_y, class_label, n_classes, fig_name="roc_auc.png"):
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#66CC99", "#999999"]
    #class_label = ["EBV", "GS", "MSI", "CIN"]
    plt.figure(figsize=(8, 8), dpi=400)
    for i in range(n_classes):
        _tmp_pred = pred_y
        _tmp_label = test_y
        _fpr, _tpr, _ = roc_curve(_tmp_label[:, i], _tmp_pred[:, i])
        _auc = auc(_fpr, _tpr)

        plt.plot(_fpr, _tpr, color=colors[i],
                 label=r'%s ROC (AUC = %0.3f)' % (class_label[i], _auc), lw=2, alpha=.9)
    plt.close('all')
    plt.style.use("ggplot")
    matplotlib.rcParams['font.family'] = "Arial"
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of HEAL Case2')
    plt.legend(loc="lower right")
    plt.savefig(fig_name, dpi=400)
    plt.close('all')


# In[25]:


'''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
'''


def plot_confusion_matrix(_model, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    plt.close('all')
    plt.figure(figsize=(8, 8), dpi=400)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix of independent test results'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix of independent test results")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.close('all')
    plt.style.use("ggplot")
    matplotlib.rcParams['font.family'] = "Arial"
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('HEAL_Workspace/figures/%s_Confusion_matrix.png' % _model, dpi=400)
    return ax


def concat_img(_model, _class_cate, _class_dict, UNIT_SIZE, col_list, row_list, y_pred, true_label, img_dirs, num_of_class=4):
    try:
    	file_name = img_dirs[0].split('/')[-3]
    except Exception as e:
        print(img_dirs)
        print(e)
    _pred_class_index = y_pred.argmax(axis=1)
    # print(_pred_class_index)
    print("Total number of tiles is %d" % len(_pred_class_index))

    voting = Counter(_pred_class_index)
    """
    _con_img = Image.new('RGBA', (UNIT_SIZE * max(col_list), UNIT_SIZE * max(row_list)), color=(255, 255, 255))
    for _i, _idx in enumerate(_pred_class_index):
        # print(_i,_idx,_pred_class_index)
        # print(img_dirs[_i])
        img = cv2.imread(img_dirs[_i])
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        red = np.full((UNIT_SIZE, UNIT_SIZE), 255)
        green = np.full((UNIT_SIZE, UNIT_SIZE), 255)
        blue = np.full((UNIT_SIZE, UNIT_SIZE), 255)

        if _idx == _class_dict[true_label]:
            tmp_r = red
            tmp_g = img_g
            tmp_b = img_b
            tmp_a = np.full((UNIT_SIZE, UNIT_SIZE), 255 * y_pred[_i, _idx])
        else:
            tmp_r = img_r
            tmp_g = img_g
            tmp_b = blue
            tmp_a = np.full((UNIT_SIZE, UNIT_SIZE), 255 * y_pred[_i, _idx])
        tmp_r = Image.fromarray(np.uint8(tmp_r), mode="L")
        tmp_g = Image.fromarray(np.uint8(tmp_g), mode="L")
        tmp_b = Image.fromarray(np.uint8(tmp_b), mode="L")
        tmp_a = Image.fromarray(np.uint8(tmp_a), mode="L")
        tmp_img = Image.merge("RGBA", (tmp_r, tmp_g, tmp_b, tmp_a))
        _con_img.paste(tmp_img, (col_list[_i] * UNIT_SIZE, row_list[_i] * UNIT_SIZE, (col_list[_i] + 1) * UNIT_SIZE,
                                 (row_list[_i] + 1) * UNIT_SIZE))

    plt.close()
    plt.title("The true label is %s" % (true_label))

    sc = plt.imshow(_con_img, vmin=0, vmax=1, cmap="RdBu_r")
    plt.colorbar(sc)
    print(voting)
    plt.savefig('HEAL_Workspace/figures/%s_%s_concat_%s_%f.png' %
                (_model, file_name, _class_cate[voting.most_common(1)[0][0]], voting.most_common(1)[0][1] / len(_pred_class_index)), dpi=500)
    """
    return voting


def model_predict(_test_loader, model_path, _class_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = nn.Sequential(model, nn.Softmax(1))
    model.eval()
    # print(model)

    pred = []
    true_label = []
    with torch.no_grad():
        for i, sample in enumerate(_test_loader, 0):
            inputs = sample["image"].to(device)
            # print(len(inputs))
            # print(sample["label"])
            labels = sample["label"].to(device)
            label_nd = []
            for _label in labels:
                l_base = [0 for i in range(_class_number)]
                l_base[_label] = 1
                label_nd.append(l_base)
            outputs = model(inputs)
            outputs = tensor2array(outputs, _class_number)
            # print(type(outputs), outputs)
            if len(pred) == 0:
                pred = outputs
                true_label = label_nd
            else:
                pred = np.vstack((pred, outputs))
                true_label = np.vstack((true_label, label_nd))
            # print(len(outputs))
            # print(pred)
            # print(true_label)
    return pred, true_label


def get_best_model(_model_name):
    _model_path = "HEAL_Workspace/models/"
    _best_model = ""
    _best_acc = 0
    for _root, _dirs, _files in os.walk(_model_path):
        for f in _files:
            tmp_model = os.path.join(_root, f)
            model_base = tmp_model.split("/")[-1].split("_")[0]
            if model_base == _model_name:
                v_acc = float((tmp_model.split("/")[-1])[:-4].split("_")[-3])
                t_acc = float((tmp_model.split("/")[-1])[:-4].split("_")[-7])
                if v_acc + t_acc >= _best_acc:
                    _best_ac = v_acc + t_acc
                    _best_model = tmp_model
    print(_best_model)
    return _best_model


def get_row_col(_img_path_df):
    _col_list = []
    _row_list = []
    # print(_img_path_df.iloc[:, 0])
    for _img_p in _img_path_df.iloc[:, 0]:
        _info = _img_p.split('/')[-1]
        _info = _info.split('.')[0]
        # print(_img_p)
        _col, _row = _info.split('_')
        _col_list.append(int(_col))
        _row_list.append(int(_row))
    return _col_list, _row_list


def independent_test(_models, _tile_size):
    conf_dict = load_variable("HEAL_Workspace/outputs/parameter.conf")
    _work_mode = conf_dict["Mode"]
    #_class_cate = list(conf_dict["Classes"])
    _class_cate = conf_dict["Classes"]
    _class_number = conf_dict["Class_number"]
    _class_idx = []
    print(_class_cate)
    for i in range(_class_number):
        _class_idx.append(i)
    _class_dict = dict(zip(_class_cate, _class_idx))

    total_pred = []
    total_label = []
    patient_pred = []
    patient_label = []

    for _model in _models:
        test_df = pd.read_csv("HEAL_Workspace/outputs/test_label_file_tiled.csv")
        # best_model = get_best_model(_model)
        best_model = "HEAL_Workspace/models/1016-092843_Vgg16_epoch_24_t-f1_0.56_t-loss_0.34_v-f1_0.27_v-loss_0.56.pkl"
        model_number = best_model.split('/')[-1].split('_')[:4]
        model_number = "_".join(str(x) for x in model_number)
        print("The tested model is %s!" % model_number)
        for i in range(len(test_df)):
            label = test_df.iloc[i, 1]
            patient_path = test_df.iloc[i, 0]
            patient_df = get_test_sample(patient_path, label)

            if len(patient_df) == 0:
                print("No testing images found under %s" % patient_path)
                continue

            test_dataset = ImageDataset(dataframe=patient_df, transform=data_transforms,
                                        class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
            val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)
            #print("starting predict...")
            _pred, _label_4d = model_predict(val_loader, best_model, _class_number)
            #print(_pred)
            col_list, row_list = get_row_col(patient_df)
            vote_res = concat_img(_model, _class_cate, _class_dict, _tile_size, col_list, row_list, _pred,
                                  label, list(patient_df.iloc[:, 0]), num_of_class=_class_number)

            _pred_p_label = _class_cate[vote_res.most_common(1)[0][0]]
            _pred_porb = vote_res.most_common(1)[0][1] / len(patient_df)

            print("Sample (%s) is predicted as %s(%f), and the true label is %s" %
                  (patient_path.split('/')[-2], _pred_p_label, _pred_porb, label))

            vect_base = [0 for n in range(_class_number)]
            for it in vote_res:
                vect_base[it] = vote_res[it] / len(patient_df)
            patient_pred.append(vect_base)

            vect_base = [0 for n in range(_class_number)]
            vect_base[_class_dict[label]] = 1
            patient_label.append(vect_base)

            if len(total_label) == 0 and len(total_pred) == 0:
                total_pred = _pred
                total_label = _label_4d
            else:
                total_pred = np.vstack((total_pred, _pred))
                total_label = np.vstack((total_label, _label_4d))
        #print(patient_pred)
        #print(patient_label)

        save_variable("HEAL_Workspace/outputs/%s_independent_tile_pred" % model_number, total_pred)
        save_variable("HEAL_Workspace/outputs/%s_independent_tile_label" % model_number, total_label)
        save_variable("HEAL_Workspace/outputs/%s_independent_patient_pred" % model_number, patient_pred)
        save_variable("HEAL_Workspace/outputs/%s_independent_patient_label" % model_number, patient_label)
        plot_roc_curve(total_pred, total_label, _class_cate, n_classes=_class_number, fig_name="HEAL_Workspace/figures/%s_roc_auc.png" % model_number)
        plot_roc_curve(np.array(patient_pred), np.array(patient_label), _class_cate, n_classes=_class_number,
                       fig_name="HEAL_Workspace/figures/%s_patient_level_roc.png" % model_number)
        plot_confusion_matrix(model_number, total_label, total_pred, _class_cate)
