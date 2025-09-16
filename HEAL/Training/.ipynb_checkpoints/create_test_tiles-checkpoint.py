import pandas as pd
import multiprocessing
import tqdm
import os
import re

def find_files(ori_folder_path, label):
    """
    Find image files in the specified folder and create a DataFrame with their paths and labels.
    
    :param ori_folder_path: Original folder path containing images.
    :param label: Label to associate with the found images.
    :return: DataFrame containing image paths and their associated label.
    """
    img_label_list = []
    folder_path = re.sub('tiling', 'tiling_macenko', ori_folder_path)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    for files in os.listdir(folder_path):
        if files.endswith(('jpeg', 'jpg')) and not files.startswith('.'):
            img_path = os.path.join(folder_path, files)
            img_label_list.append({"Image_path": img_path, "Label": label})

    return pd.DataFrame(img_label_list)

def write_test_file(test_df):
    """
    Write test files by searching for images in specified folders and saving their paths and labels.
    
    :param test_df: DataFrame containing folder paths and labels for test images.
    """
    test_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    
    def append_result(result):
        nonlocal test_img_label_df
        test_img_label_df = pd.concat([test_img_label_df, result], ignore_index=True)
        
    with multiprocessing.Pool(16) as pool:
        for idx in tqdm.tqdm(range(len(test_df))):
            label = test_df.iloc[idx, 1]
            folder_path = test_df.iloc[idx, 0]
            pool.apply_async(find_files, args=(folder_path, label), callback=append_result)
        
        pool.close()
        pool.join()
    
    test_img_label_df.to_csv("/mnt/efs-tcga/HEAL_Workspace/outputs/test_tiles.csv", encoding='utf-8', index=False)

def create_test_files():
    """
    Create test files from labeled data if the test_tiles.csv does not already exist.
    """
    if os.path.exists("/mnt/efs-tcga/HEAL_Workspace/outputs/test_tiles.csv"):
        return
    test_file_path = "/mnt/efs-tcga/HEAL_Workspace/outputs/test_label_file_tiled.csv"
    
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"The test file {test_file_path} does not exist.")
    
    wsi_df = pd.read_csv(test_file_path)
    write_test_file(wsi_df)
