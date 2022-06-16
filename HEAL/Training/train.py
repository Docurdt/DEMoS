#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""

"""

import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from skimage import io
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from HEAL.Training import model_train
from HEAL.Training import create_test_tiles

import staintools
#import pysnooper
import pip


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    install('efficientnet_pytorch')
    from efficientnet_pytorch import EfficientNet

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


def convert_label(_label, _mode, _class_num, _class_cate):
    if _mode:
        conv_label = np.zeros(_class_num)
        for _it in _label.split(','):
            for i in range(_class_num):
                if _it == str(list(_class_cate)[i]):
                    conv_label[i] = 1.0
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

    #@pysnooper.snoop()
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


def load_data(_class_cate, _class_number, _work_mode, _bs, fn = 0):
    train_data_transforms = transforms.Compose([
        transforms.Resize(size=(Image_Size, Image_Size)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    data_transforms = transforms.Compose([
        transforms.Resize(size=(Image_Size, Image_Size)),
        transforms.ToTensor()
    ])
    create_test_tiles.create_test_files()
    train_img_label_df = pd.read_csv("HEAL_Workspace/outputs/train_fold_{}.csv".format(fn))
    val_img_label_df = pd.read_csv("HEAL_Workspace/outputs/val_fold_{}.csv".format(fn))
    train_dataset = ImageDataset(dataframe=train_img_label_df, transform=train_data_transforms,
                                 class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
    val_dataset = ImageDataset(dataframe=val_img_label_df, transform=data_transforms,
                               class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
    train_loader = DataLoader(train_dataset, batch_size=_bs, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=_bs, shuffle=False, num_workers=8)
    return train_loader, val_loader


def get_model(_model_name, _class_num, _mode):
    ###resnet family###
    if _model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "ResNet18":
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "ResNet34":
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "WideResNet50":
        model = models.wide_resnet50_2(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "ResNet101":
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "WideResNet101":
        model = models.wide_resnet101_2(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "ResNet152":
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###vgg family###
    elif _model_name == "Vgg16":
        model = models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] #remove the last layer
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "Vgg16_BN":
        model = models.vgg16_bn(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] #remove the last layer
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "Vgg19":
        model = models.vgg19(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] #remove the last layer
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "Vgg19_BN":
        model = models.vgg19(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] #remove the last layer
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###alexnet###
    elif _model_name == "AlexNet":
        model = models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] #remove the last layer
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###densenet161###
    elif _model_name == "DenseNet161":
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        if _mode:
            model.classifier = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.classifier = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###InceptionV3###
    elif _model_name == "InceptionV3":
        model = models.inception_v3(pretrained=False, aux_logits = False)
        num_ftrs = model.fc.in_features
        if _mode:
            print("Using new inception_v3!!!")
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###ShuffleNetV2###
    elif _model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x2_0(pretrained=False)
        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###MobileNetV2###
    elif _model_name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        features = list(model.classifier.children())[:-1]
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###GoogleNet###
    elif _model_name == "GoogleNet":
        model = models.googlenet(pretrained=False, aux_logits = False)

        num_ftrs = model.fc.in_features
        if _mode:
            model.fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
        else:
            model.fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    ###MNASNet###
    elif _model_name == "MNASNET":
        model = models.mnasnet1_3(pretrained=False)

        num_ftrs = model.classifier[1].in_features
        features = list(model.classifier.children())[:-1]
        if _mode:
            features.extend([nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())])
            model.classifier = nn.Sequential(*features)
            criterion = nn.BCELoss()
        else:
            features.extend([nn.Linear(num_ftrs, _class_num)])
            model.classifier = nn.Sequential(*features)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion
    elif _model_name == "EFF-NET":
        model = EfficientNet.from_name('efficientnet-b1')
        for param in model.parameters():
            param.requires_grad = True

        num_ftrs = model._fc.in_features

        if _mode:
            model._fc = nn.Sequential(nn.Linear(num_ftrs, _class_num), nn.Sigmoid())
            criterion = nn.BCELoss()
        else:
            model._fc = nn.Linear(num_ftrs, _class_num)
            criterion = nn.CrossEntropyLoss()
        model = torch.nn.DataParallel(model)
        return model, criterion


def get_weight(_class_cate, _class_number, _mode):
    weights = []
    if _mode:
        for i in range(_class_number):
            weights.append(1)
    else:
        train_df = pd.read_csv("HEAL_Workspace/outputs/train_fold_0.csv")
        _class_c = list(_class_cate)
        num_count = []
        for _cc in _class_c:
            tmp_count = train_df.loc[train_df.Label == _cc]
            num_count.append(len(tmp_count))
        max_num = max(num_count)
        for _w in num_count:
            weights.append(_w/max_num)
    return weights

Image_Size = 0
def train(_models, tile_size = 512, CV_Enable = False):
    global Image_Size
    Image_Size = tile_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    conf_dict = load_variable("HEAL_Workspace/outputs/parameter.conf")
    _work_mode = conf_dict["Mode"]
    _class_cate = conf_dict["Classes"]
    _class_number = conf_dict["Class_number"]
    try:
        hp_dict = load_variable("HEAL_Workspace/outputs/hyper_parameter.conf")
        _lr = hp_dict['lr']
        _gamma = hp_dict['gamma']
        _models = [hp_dict['model_name']]
        _step_size = hp_dict['step_size']
        _bs = hp_dict['batch_size']
        print("Hyperparameters optimized by Hyperopt are using for model training, "
              "if you want to use your customized hyperparameter, please delete "
              "the configuration file: HEAL_Workspace/outputs/parameter.conf")
        print("----------------Configuration-----------------")
        print("Model architecture: {}".format(_models))
        print("Batch size: {}".format(_bs))
        print("Learning rate: {}".format(_lr))
        print("Gamma: {}".format(_gamma))
        print("Step size: {}".format(_step_size))
        print("----------------------------------------------")
    except Exception as e:
        print(e)
        print("No optimal hyper parameters found, default parameters are being used for model training ...")
        _lr = 1e-3
        _bs = 64
        _gamma = 0.4
        _step_size = 5
        print("----------------Default configuration-----------------")
        print("Model architecture: {}".format(_models))
        print("Batch size: {}".format(_bs))
        print("Learning rate: {}".format(_lr))
        print("Gamma: {}".format(_gamma))
        print("Step size: {}".format(_step_size))
        print("----------------------------------------------")

    for _model in _models:
        if(CV_Enable):
            for i in range(10):
                train_loader, val_loader = load_data(_class_cate, _class_number, _work_mode, _bs, fn=i)
                model_ft, criterion = get_model(_model, _class_number, _work_mode)
                optimizer_ft = optim.Adam(model_ft.parameters(), lr=_lr, weight_decay=1e-4)
                #exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10, 30, 50, 80], gamma=_gamma)
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, "min", factor=0.5, patience=3, threshold=1e-8)
                model_train.model_train(model_ft, _model, train_loader, val_loader, criterion,
                                        optimizer_ft, exp_lr_scheduler, _work_mode, _class_number, num_epochs=100, fn=i)
                torch.cuda.empty_cache()
        else:
            train_loader, val_loader = load_data(_class_cate, _class_number, _work_mode, _bs)
            model_ft, criterion = get_model(_model, _class_number, _work_mode)
            optimizer_ft = optim.Adam(model_ft.parameters(), lr=_lr, weight_decay=1e-4)
            #exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10, 30, 50, 80], gamma=_gamma)
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, "min", factor=0.5, patience=3, threshold=1e-8)
            model_train.model_train(model_ft, _model, train_loader, val_loader, criterion, optimizer_ft, exp_lr_scheduler, _work_mode, _class_number, num_epochs=100)
            torch.cuda.empty_cache()
