#!/usr/bin/env python3
"""
Function description:
    To tile the whole slide images into small patches of specific size on specific zoom level.

Input parameters:
    Tile_size and Tile_level

Output:
    Tiles and its path.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import tqdm
import pandas as pd
from Tiling import open_slide

import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tiling(_label_file, _test_file, _tile_size=1000, _tile_level=15):
    try:
        _train_tile_base = "/mnt/efs-tcga/HEAL_Workspace/tiling/train"
        _test_tile_base = "/mnt/efs-tcga/HEAL_Workspace/tiling/test"

        os.makedirs(_train_tile_base, exist_ok=True)
        os.makedirs(_test_tile_base, exist_ok=True)

        _format = "jpeg"
        _tile_size = _tile_size - 2
        _overlap = 1
        _limit_bounds = True
        _quality = 90
        _workers = 36
        _with_viewer = False

        if _label_file is not None:
            logging.info("Training WSIs: start tiling ...")
            _label_df = pd.read_csv(_label_file)
            _svs_path = _label_df.iloc[:, 0]
            _svs_label = _label_df.iloc[:, 1]
            new_paths = []
            new_labels = []

            for i in tqdm.tqdm(range(len(_svs_path))):
                _curr_svs = _svs_path.iloc[i]
                _curr_label = _svs_label.iloc[i]
                _folder_name = os.path.join(_train_tile_base, _curr_svs.split("/")[-1].split(".")[0])
                
                try:
                    open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                                   _tile_size, _overlap, _limit_bounds, _quality,
                                                   _workers, _with_viewer, _tile_level).run()
                    _tile_path = os.path.join(_folder_name + str("_files"), str(_tile_level))
                    new_paths.append(_tile_path)
                    new_labels.append(_curr_label)
                except Exception as e:
                    logging.error(f"Error processing {_curr_svs}: {e}")

            new_file = pd.DataFrame({"Image_path": new_paths, "Label": new_labels})
            new_file.to_csv("/mnt/efs-tcga/HEAL_Workspace/outputs/label_file_tiled.csv",
                             encoding='utf-8', index=False)
        else:
            logging.warning("No training WSI is provided.")

        if _test_file is not None:
            logging.info("Testing WSIs: start tiling ...")
            _label_df = pd.read_csv(_test_file)
            _svs_path = _label_df.iloc[:, 0]
            _svs_label = _label_df.iloc[:, 1]
            new_paths = []
            new_labels = []

            for i in tqdm.tqdm(range(len(_svs_path))):
                _curr_svs = _svs_path.iloc[i]
                _curr_label = _svs_label.iloc[i]
                _folder_name = os.path.join(_test_tile_base, _curr_svs.split("/")[-1].split(".")[0])
                
                try:
                    open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                                   _tile_size, _overlap, _limit_bounds, _quality,
                                                   _workers, _with_viewer, _tile_level).run()
                    _tile_path = os.path.join(_folder_name + str("_files"), str(_tile_level))
                    new_paths.append(_tile_path)
                    new_labels.append(_curr_label)
                except Exception as e:
                    logging.error(f"Error processing {_curr_svs}: {e}")

            new_file = pd.DataFrame({"Image_path": new_paths, "Label": new_labels})
            new_file.to_csv("/mnt/efs-tcga/HEAL_Workspace/outputs/test_label_file_tiled.csv",
                             encoding='utf-8', index=False)
        else:
            logging.warning("No testing WSI is provided.")
    except Exception as e:
        logging.critical(f"An error occurred in tiling process: {e}")
