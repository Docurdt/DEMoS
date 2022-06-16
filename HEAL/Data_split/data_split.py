#!python
# -*- coding: UTF-8 -*-

"""
Function description:
    Split the dataset into training/testing/validation;
    To generate the configuration for the model training.
"""

import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import multiprocessing
import re
from itertools import product


def save_variable(var, filename):
    pickle_f = open(filename, 'wb')
    pickle.dump(var, pickle_f)
    pickle_f.close()
    return filename


def customized_drop(_df):
    drop_idx = []
    for i in range(len(_df)):
        _ori_folder_path = _df.iloc[i, 0]
        _folder_path = re.sub('tiling', 'tiling_macenko', _ori_folder_path)
        if not os.path.exists(_folder_path):
            print(_folder_path)
            drop_idx.append(i)
        elif not os.listdir(_folder_path):
            print(_folder_path)
            drop_idx.append(i)
    _df = _df.drop(index=drop_idx)
    _df = _df.reset_index(drop = True)
    return _df


# Read csv files
def read_files(train_label_file, test_label_file):
    MLMC = False
    train_df = pd.read_csv(train_label_file)
    _tmp_label = list(train_df.iloc[:, 1])
    _class_category = []

    for _item in _tmp_label:
        _item = _item.split(",")
        for _cc in _item:
            _class_category.append(_cc)
        if len(_item) > 1:
            MLMC = True

    _class_category = list(set(list(_class_category)))
    tmp_dict = {"Mode": MLMC, "Classes": _class_category, "Class_number": len(_class_category)}
    save_variable(tmp_dict, "HEAL_Workspace/outputs/parameter.conf")

    try:
        test_df = pd.read_csv(test_label_file)
    except Exception as e:
        print("No testing label file is provided, 20% of the training dataset will be "
              "split as the testing dataset", e)
        test_df = None
    print("The original train file length is %d." % len(train_df))
    train_df = customized_drop(train_df)
    print("The wrangled train file length is %d." % len(train_df))

    if test_df is not None:
        print("The original test file length is %d." % len(test_df))
        test_df = customized_drop(test_df)
        print("The wrangled test file length is %d." % len(test_df))

    if not MLMC:
        plt.close('all')
        plt.style.use("ggplot")
        #matplotlib.rcParams['font.family'] = "Arial"
        plt.xlabel("Class")
        plt.ylabel("Number of WSIs")
        plt.title("Data distribution of the Whole Slide Images (WSIs)")
        num_class = len(_class_category)
        plt.hist(x=list(train_df.iloc[:, 1]), bins=np.arange(num_class+1)-0.5,
                 color='#0504aa', alpha=0.7, rwidth=0.5, align='mid')
        plt.grid(alpha=0.5, axis='y')
        plt.savefig("HEAL_Workspace/figures/patient_vs_class_distribution.png", dpi = 400)
        plt.clf()
        plt.cla()
        plt.close()
    return train_df, test_df, MLMC


def find_files(ori_folder_path, label):
    img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    #print(img_label_df)
    folder_path = re.sub('tiling', 'tiling_macenko', ori_folder_path)

    for files in os.listdir(folder_path):
        #print(files)
        if (files.split('.')[-1] == 'jpeg' or files.split('.')[-1] == 'jpg') and not files[0] == '.':
            img_path = os.path.join(folder_path, files)
            tmp = pd.DataFrame([[img_path, label]], columns=['Image_path', 'Label'])
            img_label_df = pd.concat([img_label_df, tmp])

    return img_label_df


result_df = pd.DataFrame(columns=['Image_path', 'Label'])
def log_result(result):
    global result_df
    result_df = pd.concat([result_df, result])


def show_tiles_distribution(df):
    #img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
    cpu_num = multiprocessing.cpu_count()
    print("The CPU number of this machine is %d" % cpu_num)
    pool = multiprocessing.Pool(int(cpu_num))
    #pool = multiprocessing.Pool(16)
    for idx in tqdm.tqdm(range(len(df))):
        label = df.iloc[idx, 1]
        folder_path = df.iloc[idx, 0]
        pool.apply_async(find_files, (folder_path, label), callback = log_result)
        #log_result(find_files(folder_path, label))
    pool.close()
    pool.join()
    plt.close('all')
    plt.style.use("ggplot")
    #matplotlib.rcParams['font.family'] = "Arial"
    plt.xlabel("Class")
    plt.ylabel("Number of tiles")
    plt.title("Data distribution of the tiles")
    num_class = len(set(list(result_df.iloc[:, 1])))
    plt.hist(x=list(result_df.iloc[:, 1]), bins=np.arange(num_class+1)-0.5,
             color='#a0040a', alpha=0.7, rwidth=0.5, align='mid')
    plt.grid(alpha=0.5, axis='y')
    plt.savefig("HEAL_Workspace/figures/tile_vs_class_distribution.png", dpi = 400)
    plt.clf()
    plt.cla()
    plt.close()


# split data using index numbers']
def split_data(train_df, test_df, test_ratio, MLMC):
    train_indices, test_indices, val_indices = [], [], []
    tmp_df = None
    if test_df == None:
        if MLMC:
            split_train_vs_test = ShuffleSplit(n_splits=1, test_size=test_ratio)
            for train_index, test_index in split_train_vs_test.split(train_df):
                train_indices = train_index
                test_indices = test_index
        else:
            split_train_vs_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio)
            for train_index, test_index in split_train_vs_test.split(train_df['Image_path'], train_df['Label']):
                train_indices = train_index
                test_indices = test_index

        tmp_df = train_df.iloc[train_indices].reset_index(drop = True)
        test_tmp_df = train_df.iloc[test_indices]
        test_tmp_df.to_csv("HEAL_Workspace/outputs/test_label_file_tiled.csv",
                           encoding='utf-8', index = False)
        if MLMC:
            split_train_vs_val = ShuffleSplit(n_splits=10, test_size=0.1)
            train_indices = []
            for a, b in split_train_vs_val.split(tmp_df):
                train_indices.append(list(a))
                val_indices.append(list(b))
        else:
            split_train_vs_val = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
            train_indices = []
            for a, b in split_train_vs_val.split(tmp_df['Image_path'], tmp_df['Label']):
                train_indices.append(list(a))
                val_indices.append(list(b))

    else:
        try:
            if MLMC:
                split_train_vs_val = ShuffleSplit(n_splits=10, test_size=0.1)
                for a, b in split_train_vs_val.split(train_df):
                    train_indices.append(list(a))
                    val_indices.append(list(b))
            else:
                split_train_vs_val = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
                for a, b in split_train_vs_val.split(train_df['Image_path'], train_df['Label']):
                    train_indices.append(list(a))
                    val_indices.append(list(b))

        except Exception as e:
            print(e)
        test_indices = None
    return tmp_df, train_indices, test_indices, val_indices


train_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])
val_img_label_df = pd.DataFrame(columns=['Image_path', 'Label'])


def write_train_file(train_df, train_indices, val_indices):

    train_val_df = pd.DataFrame(columns=['Image_path', 'Label', 'type', 'fold'])
    for i in range(10):
        train_index = train_indices[i]
        val_index = val_indices[i]
        train = train_df.iloc[train_index]
        val = train_df.iloc[val_index]
        def append_result(result):
            global train_img_label_df
            train_img_label_df = pd.concat([train_img_label_df, result])

        cpu_num = multiprocessing.cpu_count()
        print("The CPU number of this machine is %d" % cpu_num)
        pool = multiprocessing.Pool(int(cpu_num))
        #pool = multiprocessing.Pool(16)

        for idx in tqdm.tqdm(range(len(train))):
            label = train.iloc[idx, 1]
            folder_path = train.iloc[idx, 0]
            #if('F4' in folder_path.split('-')):
            pool.apply_async(find_files, args = (folder_path, label), callback = append_result)

        pool.close()
        pool.join()
        global train_img_label_df
        train_img_label_df.to_csv("HEAL_Workspace/outputs/train_fold_%d.csv" % i, encoding='utf-8', index=False)
        train_img_label_df = train_img_label_df.iloc[0:0]

        cpu_num = multiprocessing.cpu_count()
        print("The CPU number of this machine is %d" % cpu_num)
        pool = multiprocessing.Pool(int(cpu_num))
        #pool = multiprocessing.Pool(16)
        def append_result(result):
            global val_img_label_df
            val_img_label_df = pd.concat([val_img_label_df, result])
        for idx in tqdm.tqdm(range(len(val))):
            label = val.iloc[idx, 1]
            folder_path = val.iloc[idx, 0]
            pool.apply_async(find_files, args = (folder_path, label), callback = append_result)
        pool.close()
        pool.join()
        global val_img_label_df
        val_img_label_df.to_csv("HEAL_Workspace/outputs/val_fold_%d.csv" % i, encoding='utf-8', index=False)
        val_img_label_df = val_img_label_df.iloc[0:0]

def data_split(train_label_file, test_label_file=None, test_ratio=0.2):
    train_df, test_df, MLMC = read_files(train_label_file, test_label_file)
    if not MLMC:
        show_tiles_distribution(train_df)
    tmp_df, train_indices, test_indices, val_indices = split_data(train_df, test_df, test_ratio, MLMC)
    write_train_file(tmp_df, train_indices, val_indices)
