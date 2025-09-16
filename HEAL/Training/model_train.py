#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from Training.pytorchtools import EarlyStopping

# Configurações do dispositivo e TensorBoard
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="HEAL_Workspace/tfboard")
jobid = str(time.strftime('%m%d-%H%M%S', time.localtime(time.time())))

def save_variable(var, filename):
    """Salva uma variável usando pickle."""
    with open(filename, 'wb') as pickle_f:
        pickle.dump(var, pickle_f)
    return filename

def load_variable(filename):
    """Carrega uma variável usando pickle."""
    with open(filename, 'rb') as pickle_f:
        var = pickle.load(pickle_f)
    return var

def matplotlib_imshow(img, one_channel=False):
    """Exibe uma imagem usando Matplotlib."""
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # Desnormaliza
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)) if not one_channel else npimg, cmap="Greys")

class MacroSoftF1Loss(nn.Module):
    """Classe para calcular a perda F1 macro."""
    def __init__(self, consider_true_negative, sigmoid_is_applied_to_input):
        super(MacroSoftF1Loss, self).__init__()
        self._consider_true_negative = consider_true_negative
        self._sigmoid_is_applied_to_input = sigmoid_is_applied_to_input

    def forward(self, input_, target):
        target = target.float()
        input = torch.sigmoid(input_) if not self._sigmoid_is_applied_to_input else input_
        TP = torch.sum(input * target, dim=0)
        FP = torch.sum((1 - input) * target, dim=0)
        FN = torch.sum(input * (1 - target), dim=0)
        F1_class1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        loss_class1 = 1 - F1_class1

        if self._consider_true_negative:
            TN = torch.sum((1 - input) * (1 - target), dim=0)
            F1_class0 = 2 * TN / (2 * TN + FP + FN + 1e-8)
            loss_class0 = 1 - F1_class0
            loss = (loss_class0 + loss_class1) * 0.5
        else:
            loss = loss_class1

        return loss.mean()

def model_train(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, _mode, class_num, num_epochs=50, fn=0):
    """Treina o modelo especificado."""
    print(f"Model training start ({model_name}) ...")
    
    criterion2 = MacroSoftF1Loss(consider_true_negative=True, sigmoid_is_applied_to_input=False)

    training_loss, val_loss = [], []
    avg_training_loss, avg_val_loss = [], []
    patience = 5
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f"/mnt/efs-tcga/HEAL_Workspace/models/{jobid}_{model_name}_fold{fn}.pt")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        print(f"Current learning rate is {optimizer.param_groups[0]['lr']:.6f}")
        for i, sample in enumerate(train_loader):
            images = sample["image"].to(device)
            labels = sample["label"].to(device).float() if _mode else torch.nn.functional.one_hot(sample["label"], num_classes=class_num).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = (criterion(outputs, labels) + criterion2(outputs, labels)) / 2.0
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

            if i == 3:  # exemplo para visualização
                img_grid = torchvision.utils.make_grid(images)
                writer.add_image(f'Examples of training images_{i}', img_grid)
                writer.flush()

        model.eval()
        with torch.no_grad():
            for sample in val_loader:
                inputs = sample["image"].to(device)
                labels = sample["label"].to(device).float() if _mode else torch.nn.functional.one_hot(sample["label"], num_classes=class_num).to(device)
                outputs = model(inputs)
                loss = (criterion(outputs, labels) + criterion2(outputs, labels)) / 2.0
                val_loss.append(loss.item())

        training_loss_overall = torch.mean(torch.tensor(training_loss))
        val_loss_overall = torch.mean(torch.tensor(val_loss))
        scheduler.step(val_loss_overall)
        avg_training_loss.append(training_loss_overall.item())
        avg_val_loss.append(val_loss_overall.item())

        # Registro no TensorBoard
        writer.add_scalar(f'{jobid}_{model_name}_train_batch_fold{fn}/train_loss', training_loss_overall, epoch)
        writer.add_scalar(f'{jobid}_{model_name}_train_batch_fold{fn}/val_loss', val_loss_overall, epoch)
        writer.add_scalar(f'{jobid}_{model_name}_train_batch_fold{fn}/learning_rate', optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        print(f'[{epoch + 1}/{num_epochs}] train_loss: {training_loss_overall:.5f} validation_loss: {val_loss_overall:.5f}')

        # Early stopping
        early_stopping(val_loss_overall, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        training_loss.clear()
        val_loss.clear()

    # Visualização da perda
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_training_loss) + 1), avg_training_loss, label='Training Loss')
    plt.plot(range(1, len(avg_val_loss) + 1), avg_val_loss, label='Validation Loss')
    plt.axvline(avg_val_loss.index(min(avg_val_loss)) + 1, linestyle='--', color='r', label='Early Stopping Checkpoint')

   
