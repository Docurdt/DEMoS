#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import torchvision
from HEAL.Training.pytorchtools import EarlyStopping


plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="HEAL_Workspace/tfboard")
jobid = str(time.strftime('%m%d-%H%M%S', time.localtime(time.time())))


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


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class MacroSoftF1Loss(nn.Module):
    def __init__(self, consider_true_negative, sigmoid_is_applied_to_input):
        super(MacroSoftF1Loss, self).__init__()
        self._consider_true_negative = consider_true_negative
        self._sigmoid_is_applied_to_input = sigmoid_is_applied_to_input

    def forward(self, input_, target):
        target = target.float()
        if self._sigmoid_is_applied_to_input:
            input = input_
        else:
            input = torch.sigmoid(input_)
        TP = torch.sum(input * target, dim=0)
        FP = torch.sum((1 - input) * target, dim=0)
        FN = torch.sum(input * (1 - target), dim=0)
        F1_class1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        loss_class1 = 1 - F1_class1
        if self._consider_true_negative:
            TN = torch.sum((1 - input) * (1 - target), dim=0)
            F1_class0 = 2*TN/(2*TN + FP + FN + 1e-8)
            loss_class0 = 1 - F1_class0
            loss = (loss_class0 + loss_class1)*0.5
        else:
            loss = loss_class1
        macro_loss = loss.mean()
        return macro_loss


def model_train(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, _mode, class_num, num_epochs=50, fn = 0):
    print("Model training start (%s) ..." % model_name)
    
    criterion2 = MacroSoftF1Loss(consider_true_negative=True, sigmoid_is_applied_to_input=False)

    training_loss = []
    val_loss = []
    avg_training_loss = []
    avg_val_loss = []
    # initialize the early_stopping object
    patience = 5
    early_stopping = EarlyStopping(patience=patience, verbose=True, path="HEAL_Workspace/models/%s_%s_fold%d.pt" % (jobid, model_name, fn))
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        print("Current learning rate is %f" % optimizer.param_groups[0]["lr"])
        for i, sample in enumerate(train_loader, 0):
            #print("The input is %s"%sample["image"])
            if(i == 3):
                # create grid of images
                images = sample["image"]
                img_grid = torchvision.utils.make_grid(images)
                # show images
                # matplotlib_imshow(img_grid, one_channel=False)
                # write to tensorboard
                writer.add_image('Examples of training images_%d' % i, img_grid)
                writer.flush()
                
#             if(i == 100):
#                 break

            inputs = sample["image"].to(device)
            if _mode:
                labels = sample["label"].to(device).float()
            else:
                labels = sample["label"].to(device)
                labels_oh = torch.nn.functional.one_hot(sample["label"], num_classes=class_num).to(device)
            #print("Label is %s"%labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = (criterion(outputs, labels) + criterion2(outputs, labels_oh))/2.0
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

########################################
        #print(model)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #time.sleep(30)
###################################
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(val_loader, 0):
                inputs = sample["image"].to(device)
                if _mode:
                    labels = sample["label"].to(device).float()
                else:
                    labels = sample["label"].to(device)
                    labels_oh = torch.nn.functional.one_hot(sample["label"], num_classes=class_num).to(device)
                outputs = model(inputs)
                loss = (criterion(outputs, labels) + criterion2(outputs, labels_oh))/2.0
                val_loss.append(loss.item())
        
        training_loss_overall = np.average(training_loss)
        val_loss_overall = np.average(val_loss)
        scheduler.step(val_loss_overall)
        avg_training_loss.append(training_loss_overall)
        avg_val_loss.append(val_loss_overall)

        writer.add_scalar('%s_%s_train_batch_fold%d/train_loss' % (jobid, model_name, fn), training_loss_overall, epoch)
        writer.add_scalar('%s_%s_train_batch_fold%d/val_loss' % (jobid, model_name, fn), val_loss_overall, epoch)
        writer.add_scalar('%s_%s_train_batch_fold%d/learning_rate' % (jobid, model_name, fn), optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        epoch_len = len(str(num_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {training_loss_overall:.5f} ' +
                     f'validation_loss: {val_loss_overall:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        training_loss = []
        val_loss = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss_overall, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    #     # load the last checkpoint with the best model
    # model.load_state_dict(torch.load('checkpoint.pt'))
    #
    # torch.save(model.state_dict(), "HEAL_Workspace/models/%s_%s_fold%d.pt" % (jobid, model_name, fn))

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_training_loss) + 1), avg_training_loss, label='Training Loss')
    plt.plot(range(1, len(avg_val_loss) + 1), avg_val_loss, label='Validation Loss')

    # find position of lowest validation loss
    min_pos = avg_val_loss.index(min(avg_val_loss)) + 1
    plt.axvline(min_pos, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2.5)  # consistent scale
    plt.xlim(0, len(avg_training_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig('HEAL_Workspace/figures/%s_%s_loss_fold%d_plot.png' % (jobid, model_name, fn), bbox_inches='tight')

    print('%s Training finished!' % model_name)
    return model
