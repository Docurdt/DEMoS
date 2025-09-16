#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle
import csv
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
import pathlib
import os


SAMPLE_SIZE = 0.2
TEST_SIZE = 1000
TEST_EPOCHS = 10



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


train_data_transforms = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomRotation(5),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    #transforms.Normalize([0.7070, 0.5345, 0.6812], [0.1649, 0.2027, 0.1498])
])
data_transforms = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor()
    #transforms.Normalize([0.7070, 0.5345, 0.6812], [0.1649, 0.2027, 0.1498])
])


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


def load_data(_class_cate, _class_number, _work_mode, bs):
    train_img_label_df = pd.read_csv("HEAL_Workspace/outputs/train_fold_0.csv")
    val_img_label_df = pd.read_csv("HEAL_Workspace/outputs/val_fold_0.csv")
    train_dataset = ImageDataset(dataframe=train_img_label_df, transform=train_data_transforms,
                                 class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
    val_dataset = ImageDataset(dataframe=val_img_label_df, transform=data_transforms,
                               class_cate=_class_cate, class_num=_class_number, mode=_work_mode)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
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


def train(model, optimizer, train_loader, criterion, scheduler, _mode, device):
    model.train()
    for epoch in range(TEST_EPOCHS):
        for i, sample in enumerate(train_loader, 0):
            if i * len(sample) > SAMPLE_SIZE * len(train_loader.dataset):
                break
            inputs = sample["image"].to(device)
            if _mode:
                labels = sample["label"].to(device).float()
            else:
                labels = sample["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return


def test(model, val_loader, criterion, _mode, device):
    model.eval()
    running_loss = 0.0
    n_total = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader, 0):
            if i * len(sample) > TEST_SIZE:
                break
            inputs = sample["image"].to(device)
            if _mode:
                labels = sample["label"].to(device).float()
            else:
                labels = sample["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            n_total += 1
    return running_loss/n_total


class HyperoptTrain:
    def __init__(self, model_name, learning_rate, step_size, _gamma, batch_size):
        try:
            torch.cuda.empty_cache()
        except:
            print("Failed to clean the GPU cache.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_dict = load_variable("HEAL_Workspace/outputs/parameter.conf")
        self._work_mode = self.conf_dict["Mode"]
        self._class_cate = self.conf_dict["Classes"]
        self._class_number = self.conf_dict["Class_number"]
        self.train_loader, self.val_loader = load_data(self._class_cate, self._class_number, self._work_mode, batch_size)
        self.model_ft, self.criterion = get_model(model_name, self._class_number, self._work_mode)
        self.model_ft = self.model_ft.to(self.device)
        self.optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=learning_rate)
        self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer_ft, step_size=step_size, gamma=_gamma)

    def train_model(self):
        train(self.model_ft, self.optimizer_ft, self.train_loader, self.criterion, self.exp_lr_scheduler, self._work_mode, self.device)
        val_loss = test(self.model_ft, self.val_loader, self.criterion, self._work_mode, self.device)
        return val_loss
