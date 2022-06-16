#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
from HEAL.HEAL_APP import tiling
from HEAL.HEAL_APP import train
from HEAL.HEAL_APP import cross_validation
from HEAL.HEAL_APP import independent_test
from HEAL.HEAL_APP import survival_analysis

"""
import pickle
from HEAL.Tiling import tiling
from HEAL.Pre_processing import pre_processing
from HEAL.Data_split import data_split
from HEAL.Training import train
from HEAL.Independent_test import independent_test
from HEAL.Hyperparameter_optimisation import hyperparameter_optimisation
from HEAL.Data_visualisation import data_visualisation
from HEAL.Grad_Cam import grad_cam


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


def run(**kwargs):
    """
    This is the main entry point of the HEAL package.
    Default parameters:
        label_file = "label_file.csv";
        testing_label_file = None;
        models = ['ResNet50'];
        training_mode = "single_round";
        procedure = ["Tiling", "Data_Split", "Training",  "Testing", "Survival_analysis"].

    **parameter description**
        1) label_file: (string) the file path of the input csv format file. The csv file
        contains two columns,
            col_1: "Image_path" (the image file paths of each patient)
            col_2: "Label" (the label of the corresponding patient, each sample could
                contains several labels.)
        2) testing_label_file: (string) the file path of the input csv format file for
         model training.
            This parameter is None as default, the testing dataset will be split from
            the "label file".
            If the user input test file, this file will be used for independent test.
        3) models: (list) specify the deep learning models used for model training.
            The available options are: VGG11, VGG19, ResNet36, ResNet50, Inception-V3,
            SENet, ResNeXt ...;
        4) training_mode: (string) specify the training mode, single round training or
        10-fold cross-validation;
        5) procedure: (list) specify the detailed processing steps for the model training.
        The entire processing steps is consisted by 5 steps:
                a) tiling: segment the whole slide image into small patches, the default
                image size is 1000*1000;
                b) data_split: split the dataset into training dataset, validation dataset,
                and testing dataset;
                c) training: to train the model based on the input images and the specified
                models;
                d) testing: to test the model performance based on the optimised model;
                e) survival_analysis: (Optional) to conduct the survival analysis based on
                 the output of the independent test.

    """
    # print(kwargs.items())
    # Parse the input parameters
    _label_file = None
    _testing_label_file = None
    _models = None
    _training_mode = None
    _procedure = None
    _tile_info = None
    _filter_model = None
    _extra_test_file = None
    _extra_testing_pre_processing_enable = False
    for _key, _val in kwargs.items():
        # print(_key, _val)
        if _key is "label_file":
            _label_file = _val
        elif _key is "tile_info":
            _tile_info = _val
        elif _key is "filter_model":
            _filter_model = _val
        elif _key is "testing_label_file":
            _testing_label_file = _val
        elif _key is "extra_testing_label_file":
            _extra_test_file = _val
        elif _key is "extra_testing_pre_processing_enable":
            _extra_testing_pre_processing_enable = _val
        elif _key is "models":
            _models = _val
        elif _key is "training_mode":
            _training_mode = _val
        elif _key is "procedure":
            _procedure = _val
    _label_file_tiled = None
    if _label_file is not None:
        _label_file_tiled = "HEAL_Workspace/outputs/label_file_tiled.csv"

    _test_label_file_tiled = None
    if _testing_label_file is not None:
        _test_label_file_tiled = "HEAL_Workspace/outputs/test_label_file_tiled.csv"

    # Call the corresponding functions based on the specified procedures.
    for _proc in _procedure:
        if _proc is "Tiling":
            print("[INFO] Start tiling ...")
            tiling.tiling(_label_file, _testing_label_file, _tile_size=_tile_info[0], _tile_level=_tile_info[1])
        elif _proc is "Pre_processing":
            print("[INFO] Image pre-processing: color correction and blur detection ...")
            pre_processing.pre_processing()
        elif _proc is "Data_split":
            print("[INFO] Data split ...")
            data_split.data_split(_label_file_tiled, _test_label_file_tiled)
        elif _proc is "Hyperparameter_optimisation":
            print("[INFO] Using HyperOpt to optimise the parameters ...")
            hyperparameter_optimisation.tuning()
        elif _proc is "Training":
            if _training_mode is "Single_round":
                print("[INFO] Training the model in single round mode ...")
                train.train(_models, tile_size=_tile_info[0])
            elif _training_mode is "Cross_validation":
                print("[INFO] Training the model in 10-fold cross-validation mode ...")
                train.train(_models, tile_size=_tile_info[0], CV_Enable=True)
        elif _proc is "Testing":
            print("[INFO] Running the independent test ...")
            independent_test.independent_test(_models, _tile_info, extra_test_set=_extra_test_file,
                                              pre_processing_enable=_extra_testing_pre_processing_enable)
        elif _proc is "Data_visualisation":
            print("[INFO] Running the data visualisation ...")
            data_visualisation.data_visualisation(_tile_info)
        elif _proc is "Grad_CAM":
            print("[INFO] Using Grad-CAM to visualize the key regions ...")
            grad_cam.grad_cam()

    print(_label_file, _testing_label_file, _models, _training_mode, _procedure)
