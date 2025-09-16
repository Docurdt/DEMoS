import sys
import os
import pickle
from Tiling import tiling
from Pre_processing import pre_processing
from Data_split import data_split
from Training import train
from Independent_test import independent_test
from Hyperparameter_optimisation import hyperparameter_optimisation
from Data_visualisation import data_visualisation
from Grad_Cam import grad_cam


def save_variable(var, filename):
    with open(filename, 'wb') as pickle_f:
        pickle.dump(var, pickle_f)
    return filename


def load_variable(filename):
    with open(filename, 'rb') as pickle_f:
        var = pickle.load(pickle_f)
    return var
import argparse
import os

def run(**kwargs):
    """
    This is the main entry point of the HEAL package.
    Default parameters:
        label_file = "label_file.csv";
        testing_label_file = None;
        models = ['ResNet50'];
        training_mode = "single_round";
        procedure = ["Tiling", "Pre_processing", "Data_Split", "Training",  "Hyperparameter_optimisation", "Testing", "Survival_analysis"].

    **Parameter description**
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
    print(kwargs.items())

    # Parse the input parameters
    _label_file = kwargs.get("label_file", None)
    _testing_label_file = kwargs.get("testing_label_file", None)
    _models = kwargs.get("models", None)
    _training_mode = kwargs.get("training_mode", "Single_round")
    _procedure = kwargs.get("procedure", None)
    _tile_info = kwargs.get("tile_info", [256, 0])  # Default values for tile size and level
    _filter_model = kwargs.get("filter_model", None)
    _extra_test_file = kwargs.get("extra_testing_label_file", None)
    _extra_testing_pre_processing_enable = kwargs.get("extra_testing_pre_processing_enable", False)

    _label_file_tiled = "/mnt/efs-tcga/HEAL_Workspace/outputs/label_file_tiled.csv" if _label_file else None
    _test_label_file_tiled = "/mnt/efs-tcga/HEAL_Workspace/outputs/test_label_file_tiled.csv" if _testing_label_file else None

    # Create necessary directories if they do not exist
    os.makedirs('/mnt/efs-tcga/HEAL_Workspace/tiling/train', exist_ok=True)
    os.makedirs('/mnt/efs-tcga/HEAL_Workspace/outputs', exist_ok=True)

    # Call the corresponding functions based on the specified procedures
    for _proc in _procedure:
        if _proc == "Tiling":
            print("[INFO] Start tiling ...")
            tiling.tiling(_label_file, _testing_label_file, _tile_size=_tile_info[0], _tile_level=_tile_info[1])
        elif _proc == "Pre_processing":
            print("[INFO] Image pre-processing: color correction and blur detection ...")
            _img_path_template = input("Please input the path of the template image: ")
            pre_processing.pre_processing(_img_path_template=_img_path_template)
        elif _proc == "Data_split":
            print("[INFO] Data split ...")
            data_split.data_split(_label_file_tiled, _test_label_file_tiled)
        elif _proc == "Hyperparameter_optimisation":
            print("[INFO] Using HyperOpt to optimise the parameters ...")
            hyperparameter_optimisation.tuning()
        elif _proc == "Training":
            if _training_mode == "Single_round":
                print("[INFO] Training the model in single round mode ...")
                train.train(_models, tile_size=_tile_info[0])
            elif _training_mode == "Cross_validation":
                print("[INFO] Training the model in 10-fold cross-validation mode ...")
                train.train(_models, tile_size=_tile_info[0], CV_Enable=True)
        elif _proc == "Testing":
            print("[INFO] Running the independent test ...")
            independent_test.independent_test(_models, _tile_info, extra_test_set=_extra_test_file,
                                              pre_processing_enable=_extra_testing_pre_processing_enable)
        elif _proc == "Data_visualisation":
            print("[INFO] Running the data visualisation ...")
            data_visualisation.data_visualisation(_tile_info)
        elif _proc == "Grad_CAM":
            print("[INFO] Using Grad-CAM to visualize the key regions ...")
            grad_cam.grad_cam()

    print(_label_file, _testing_label_file, _models, _training_mode, _procedure)


def main():
    parser = argparse.ArgumentParser(description="Run the HEAL package pipeline for image analysis and model training.")
    
    parser.add_argument('--label_file', type=str, required=True, 
                        help="(required) Path to the label file containing the training data.")
    parser.add_argument('--testing_label_file', type=str, default=None, 
                        help="(optional) Path to the testing label file. If not provided, the main label file will be used for testing.")
    parser.add_argument('--models', type=str, nargs='+', required=True, 
                        help=("(required) List of models to use for training and testing:"
                                "AlexNet, DenseNet161, EFF-NET, GoogleNet, InceptionV3, MNASNET, "
                                "MobileNetV2, ResNet101, ResNet152, ResNet18, ResNet34, ResNet50, "
                                "ShuffleNetV2, Vgg16, Vgg16_BN, Vgg19, Vgg19_BN, WideResNet101, "
                                "WideResNet50"
                            ))
    parser.add_argument('--training_mode', type=str, choices=['Single_round', 'Cross_validation'], default='Single_round',
                        help="(optional) Training mode, either single_round or Cross_validation. Default is single_round.")
    parser.add_argument('--procedure', type=str, nargs='+', required=True, 
                        help=("(required) Procedure steps to run, specified as a list of strings:"
                                "Tiling: Divides the images into smaller tiles for processing."
                                "Pre_processing: Prepares the images for analysis."
                                "Data_split: Splits the data into training, validation, and testing sets."
                                "Hyperparameter_optimisation: Optimises the hyperparameters of the model."
                                "Training: Trains the model using the training data."
                                "Testing: Tests the model using the testing data."
                                "Data_visualisation: Visualises the processed data."
                                "Grad_CAM: Uses Grad-CAM to visualise the key regions of the image."
                            ))
    parser.add_argument('--tile_info', type=int, nargs=2, default=[1000, 0], 
                        help="(optional) Tile size and level, specified as two integers. Default is [1000, 0].")
    parser.add_argument('--filter_model', type=str, default=None, 
                        help="(optional) Model filtering criteria.")
    parser.add_argument('--extra_testing_label_file', type=str, default=None, 
                        help="(optional) Path to an additional testing label file.")
    parser.add_argument('--extra_testing_pre_processing_enable', action='store_true', 
                        help="(optional) Whether to enable pre-processing for the extra testing set.")
    
    args = parser.parse_args()
    
    run(
        label_file=args.label_file,
        testing_label_file=args.testing_label_file,
        models=args.models,
        training_mode=args.training_mode,
        procedure=args.procedure,
        tile_info=args.tile_info,
        filter_model=args.filter_model,
        extra_testing_label_file=args.extra_testing_label_file,
        extra_testing_pre_processing_enable=args.extra_testing_pre_processing_enable
    )

if __name__ == "__main__":
    main()