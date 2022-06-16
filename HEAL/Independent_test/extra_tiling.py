#!/usr/bin/env python3
"""
Function description:
    To tile the extra independent test whole slide images into small patches of specific size on specific zoom level.

Input parameters:
    Tile_size and Tile_level

Output:
    Tiles and its path.
"""
import os
import tqdm
import pandas as pd
from HEAL.Tiling import open_slide

def extra_tiling(_extra_test_file, _tile_size=1000, _tile_level=15):
    """
    Tile the WSIs into patches for training dataset and testing dataset.
    If the testing dataset existed, put the testing tiles into the test folder;
    If not, the test dataset will be split from the training dataset.
    :param _label_file:
    :param _test_file:
    :param _tile_size:
    :param _tile_level:
    :return: folder path of all the tiled images and its labels
    """
    _train_tile_base = "HEAL_Workspace/tiling/train"
    _test_tile_base = "HEAL_Workspace/tiling/extra_test"

    if not os.path.exists(_train_tile_base):
        os.mkdir(_train_tile_base)
    if not os.path.exists(_test_tile_base):
        os.mkdir(_test_tile_base)

    _format = "jpeg"
    _tile_size = _tile_size - 2
    _overlap = 1
    _limit_bounds = True
    _quality = 90
    _workers = 36
    _with_viewer = False

    if _extra_test_file is not None:
        print("Extra Testing WSIs: start tiling ...")
        _label_df = pd.read_csv(_extra_test_file)
        _svs_path = _label_df.iloc[:, 0]
        _svs_label = _label_df.iloc[:, 1]
        new_path = []
        new_label = []
        for i in tqdm.tqdm(range(len(_svs_path))):
            _curr_svs = _svs_path.iloc[i]
            _curr_label = _svs_label.iloc[i]
            _folder_name = os.path.join(_test_tile_base, _curr_svs.split("/")[-1].split(".")[0])
            open_slide.DeepZoomStaticTiler(_curr_svs, _folder_name, _format,
                                           _tile_size, _overlap, _limit_bounds, _quality,
                                           _workers, _with_viewer, _tile_level).run()
            _tile_path = os.path.join(_folder_name+str("_files"), str(_tile_level))
            # if os.path.exists(_tile_path):
            new_path.append(_tile_path)
            new_label.append(_curr_label)
        tmp_dict = {"Image_path": new_path, "Label": new_label}
        new_file = pd.DataFrame(tmp_dict)
        new_file.to_csv("HEAL_Workspace/outputs/extra_test_label_file_tiled.csv",
                        encoding='utf-8', index=False)
