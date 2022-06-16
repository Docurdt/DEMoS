#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""

"""

import os
import pickle
import re
from collections import Counter
import cv2
import matplotlib
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
import time
from HEAL.Independent_test import extra_tiling
from HEAL.Pre_processing import pre_processing
from HEAL.Independent_test import create_test_file
from HEAL.Training import train

plt.ion()   # interactive mode

def save_variable(var, filename):
    pickle_f = open(filename, 'wb')
    pickle.dump(var, pickle_f)
    pickle_f.close()
    return filename


def load_variable(filename):
    pickle_f = open(filename, 'rb')
    var = pickle.load(pickle_f)
    pickle_f.close()
    return var


def get_models():
    model_path = "HEAL_Workspace/models"
    model_list = []
    for _r, _dirs, _fs in os.walk(model_path):
        for _f in _fs:
            if re.match(r'^\d*.', str(_f)) and str(_f).endswith('pt'):
                model_list.append(os.path.join(_r, _f))
    return model_list


def convert_label(_label, _mode, _class_num, _class_cate):
    conv_label = np.zeros(_class_num)
    if _mode:
        for _it in _label.split(','):
            for i in range(_class_num):
                if _it == str(list(_class_cate)[i]):
                    conv_label[i] = 1.0
        return conv_label
    else:
        for i in range(_class_num):
            if str(_label) == list(_class_cate)[i]:
                conv_label[i] = 1.0
        return conv_label


class ImageDataset(Dataset):
    def __init__(self, dataframe=None, transform=None, class_cate=None, class_num=None, mode=None):
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
        sample = {"image": img, "label": _label, "image_path": img_path}
        return sample


def get_test_sample(_patient, _label):
    _test_img_df = pd.DataFrame(columns=['Image_path', 'Label'])
    for _files in os.listdir(_patient):
        if _files.split('.')[-1] == 'jpeg':
            _img_path = os.path.join(_patient, _files)
            _tmp = pd.DataFrame([[_img_path, _label]], columns=['Image_path', 'Label'])
            _test_img_df = pd.concat([_test_img_df, _tmp])
    return _test_img_df


def model_predict(_model, model_path, _test_loader, _class_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.load_state_dict(torch.load(model_path, map_location=device))
    _model.to(device)
    _model.eval()
    # print(model)

    pred = np.array([])
    true_label = np.array([])
    sample_path = []
    with torch.no_grad():
        for i, sample in enumerate(_test_loader, 0):
            inputs = sample["image"].to(device)
            print(inputs)
            # print(sample["label"])
            labels = sample["label"].to(device)
            label_nd = labels
            outputs = _model(inputs)
            pred = np.vstack([pred, outputs.cpu().detach().numpy()]) if pred.size else outputs.cpu().detach().numpy()
            true_label = np.vstack([true_label, label_nd.cpu()]) if true_label.size else label_nd.cpu()
            sample_path = sample_path + sample["image_path"]
            # print(len(outputs))
            # print(pred)
            # print(true_label)
    return pred, true_label, sample_path


def independent_test(_models, _tile_info, extra_test_set=None, pre_processing_enable=False):
    _tile_size = _tile_info[0]
    _tile_level = _tile_info[1]
    conf_dict = load_variable("HEAL_Workspace/outputs/parameter.conf")
    _work_mode = conf_dict["Mode"]
    _class_cate = list(conf_dict["Classes"])
    print(_class_cate)
    print(extra_test_set, pre_processing_enable)
    _class_number = conf_dict["Class_number"]
    _class_idx = []
    for i in range(_class_number):
        _class_idx.append(i)
    _class_dict = dict(zip(_class_cate, _class_idx))

    final_test_file = None
    if extra_test_set:
        extra_tiling.extra_tiling(extra_test_set, _tile_size=_tile_size, _tile_level=_tile_level)
        final_test_file = "HEAL_Workspace/outputs/extra_test_label_file_tiled.csv"
        final_test_file = extra_test_set
        if pre_processing_enable:
            if not os.path.exists("HEAL_Workspace/tiling_macenko/extra_test"):
                print("Pre-processing for external data")
                pre_processing.pre_processing(extra_prefix="/extra_test")
            ori_e_test_file = final_test_file
            dst_e_test_file = "HEAL_Workspace/outputs/extra_test_file_preprocessed.csv"
            pre_flag = create_test_file.create_test_files(ori_test_file=ori_e_test_file, dst_test_file=dst_e_test_file)
            print(pre_flag)
            if pre_flag:
                final_test_file = dst_e_test_file

    else:
        final_test_file = "HEAL_Workspace/outputs/test_label_file_tiled.csv"
        macenko_test_file = "HEAL_Workspace/outputs/test_file_preprocessed.csv"
        pre_flag = create_test_file.create_test_files(ori_test_file=final_test_file, dst_test_file=macenko_test_file)
        if pre_flag:
            final_test_file = macenko_test_file

    if os.path.exists(final_test_file):
        test_df = pd.read_csv(final_test_file)
        data_transforms = transforms.Compose([transforms.Resize(size=(_tile_size, _tile_size)), transforms.ToTensor()])
        test_dataset = ImageDataset(dataframe=test_df, transform=data_transforms,
                                    class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

        model_paths = get_models()
        for tmp_model in model_paths:
            model_base = tmp_model.split("_")[2]
            print(model_base)
            t_model, criterion = train.get_model(model_base, _class_number, _work_mode)
            preds, labels, paths = model_predict(t_model, tmp_model, test_loader, _class_number)

            results_record_path = "HEAL_Workspace/outputs/Results_EX" + str(tmp_model.split("/")[-1]).split(".")[0] + "_" +\
                                  str(final_test_file.split("/")[-1]).split(".")[0] + ".out"
            tmp_results = {"preds": preds, "labels": labels, "sample_path": paths}
            save_variable(tmp_results, results_record_path)
            print("Testing result has been saved as \"" + results_record_path + "\"")

        #plot roc_curve()
        #plot confusion_matrix()
        #plot CONC_HEATMAP

    else:
        print("[Error] Testing label file is not existing")

