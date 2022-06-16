import pandas as pd
import multiprocessing
import tqdm
import os
import re


test_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
def write_train_file(test_df):
    def append_result(result):
        global test_img_label_df
        test_img_label_df = pd.concat([test_img_label_df, result])
    pool = multiprocessing.Pool(16)
    for idx in tqdm.tqdm(range(len(test_df))):
        label = test_df.iloc[idx, 1]
        folder_path = test_df.iloc[idx, 0]
        pool.apply_async(find_files, args = (folder_path, label), callback = append_result)
    pool.close()
    pool.join()
    global train_img_label_df
    test_img_label_df.to_csv("HEAL_Workspace/outputs/test_tiles.csv", encoding='utf-8', index=False)


def find_files(ori_folder_path, label):
    img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    folder_path = re.sub('tiling', 'tiling_macenko', ori_folder_path)
    for files in os.listdir(folder_path):
        if (files.split('.')[-1] == 'jpeg' or files.split('.')[-1] == 'jpg') and not files[0] == '.':
            img_path = os.path.join(folder_path, files)
            tmp = pd.DataFrame([[img_path, label]], columns=['Image_path', 'Label'])
            img_label_df = pd.concat([img_label_df, tmp])
    return img_label_df


def create_test_files():
    if os.path.exists("HEAL_Workspace/outputs/test_tiles.csv"):
        return
    else:
        test_file_path = "HEAL_Workspace/outputs/test_label_file_tiled.csv"
        wsi_df = pd.read_csv(test_file_path)
        write_train_file(wsi_df)
