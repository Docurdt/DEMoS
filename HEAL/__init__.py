#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
HEAL is a deep learning-based universal H&E stained image analysis pipeline.\
With a GPU equipped desktop computer, the users could build their deep learning \
models based on their own in-house image dataset, and conduct the subsequent\
survival analysis.

Function:
    Initialize the workspace of the HEAL pipeline.
Inputs:
    N.A.
Outputs:
    N.A.
"""
import os

__all__ = ['Tiling', 'Pre_processing', 'Data_split', 'Hyperparameter_optimisation', 'Training', 'Data_visualisation',
           'Independent_test', 'Grad_Cam', 'run']

if __name__ == "__main__":
    print("HEAL is a universal H/E stained slide image analysis pipeline!")
else:
    # Define all the modules and packages of HEAL.

    # This folder is the workspace for HEAL;
    if not os.path.exists("HEAL_Workspace/"):
        os.mkdir("HEAL_Workspace/")
        print("\"Workspace\" for HEAL created.")
    # This folder is used to store all the tiles;
    if not os.path.exists("HEAL_Workspace/tiling"):
        os.mkdir("HEAL_Workspace/tiling")
        print("\"tiling\" folder created.")
    # This folder is used to store the outputs files during the pipeline running;
    if not os.path.exists("HEAL_Workspace/outputs"):
        os.mkdir("HEAL_Workspace/outputs")
        print("\"output\" folder created.")
    # This folder is used to store the deep learning models;
    if not os.path.exists("HEAL_Workspace/models"):
        os.mkdir("HEAL_Workspace/models")
        print("\"models\" folder created.")
    # This folder is used to store the logs;
    if not os.path.exists("HEAL_Workspace/logs"):
        os.mkdir("HEAL_Workspace/logs")
        print("\"logs\" folder created.")
    # This folder is used to store the tensorboard files for the data visualization.
    if not os.path.exists("HEAL_Workspace/tfboard"):
        os.mkdir("HEAL_Workspace/tfboard")
        print("\"tfboard\" folder created.")
    # This folder is used to store the generated figures.
    if not os.path.exists("HEAL_Workspace/figures"):
        os.mkdir("HEAL_Workspace/figures")
        print("\"figures\" folder created.")
