import pandas as pd
import multiprocessing
import tqdm
import os
import re


test_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
def write_train_file(test_df, dst_file):
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
    test_img_label_df.to_csv(dst_file, encoding='utf-8', index=False)


def find_files(ori_folder_path, label):
    img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    folder_path = re.sub('tiling', 'tiling_macenko', ori_folder_path)
    for files in os.listdir(folder_path):
        if (files.split('.')[-1] == 'jpeg' or files.split('.')[-1] == 'jpg' or files.split('.')[-1] == 'png') and not files[0] == '.':
            img_path = os.path.join(folder_path, files)
            tmp = pd.DataFrame([[img_path, label]], columns=['Image_path', 'Label'])
            img_label_df = pd.concat([img_label_df, tmp])
    return img_label_df


def create_test_files(ori_test_file, dst_test_file):
    if os.path.exists("HEAL_Workspace/tiling_macenko"):
        if os.path.exists(dst_test_file):
            return True
        else:
            wsi_df = pd.read_csv(ori_test_file)
            write_train_file(wsi_df, dst_test_file)
            return True
    else:
        return False
