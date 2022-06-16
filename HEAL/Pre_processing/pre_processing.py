import staintools
from pathlib import Path
import pandas as pd
import re
import os
import cv2 as cv
from shutil import copy
from matplotlib import pyplot as plt
import tqdm
import multiprocessing
from pathlib import Path

'''
Function description:
    Detect the blurred images and move them to a backup directory;
    Correct all the clear images into a unified color space;
'''


def image_convert(_img_path, _new_img_path):
    template_image = Path("HEAL/Pre_processing/n6.png")
    template_image = str(template_image.resolve())
    target = staintools.read_image(template_image)
    #target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(target)
    #print(normalizer.stain_matrix_target)
    image = staintools.read_image(_img_path)
    image = staintools.LuminosityStandardizer.standardize(image)
    img = normalizer.transform(image)
    cv.imwrite(_new_img_path, img)


def create_new_folder(_patient_path, ori_str, replace_str):
    _new_patient_path = re.sub(ori_str, replace_str, _patient_path)
    print(_new_patient_path)
    Path(_new_patient_path).mkdir(parents=True, exist_ok=True)
    return (_new_patient_path)


def variance_of_laplacian(image):
    #     compute the Laplacian of the image and then return the focus
    #     measure, which is simply the variance of the Laplacian
    return cv.Laplacian(image, cv.CV_64F).var()


def find_blur(imagePath):
    imagePath = Path(imagePath)
    imagePath = imagePath.resolve()
    image = cv.imread(str(imagePath))

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return image, fm


def generate_blur_score(input_csv):
    input_df = pd.read_csv(input_csv)
    output_df = pd.DataFrame(columns=['Image_path', 'Label', 'Blur_score'])

    out_csv = input_csv.split(".")[0] + "_blur_bak.csv"
    print("output csv is: %s" % out_csv)

    for i in range(len(input_df)):
        _img_path = input_df['Image_path'][i]
        _label = input_df['Label'][i]
        img, fm = find_blur(_img_path)
        _tmp = pd.DataFrame([[_img_path, _label, fm]], columns=['Image_path', 'Label', 'Blur_score'])
        output_df = pd.concat([output_df, _tmp])
    output_df.to_csv(out_csv)


def blur_color_processing(_root, _img_path, _img):
    print(_img_path)
    img, fm = find_blur(_img_path)
    print(fm)
    if fm <= 100:
        blur_path = create_new_folder(_root, "tiling", "tiling_blur")
        copy(_img_path, blur_path)
    else:
        _new_img_folder = create_new_folder(_root, "tiling", "tiling_macenko")
        _new_img_path = os.path.join(_new_img_folder, _img)
        image_convert(_img_path, _new_img_path)
    return (fm)


def pre_processing(extra_prefix=""):
    print("[INFO] Starting blur detection ...")
    cpu_num = multiprocessing.cpu_count()
    print("The CPU number of this machine is %d" % cpu_num)
    pool = multiprocessing.Pool(int(cpu_num))
    _image_path = "HEAL_Workspace/tiling" + str(extra_prefix)
    for _root, _dir, _imgs in os.walk(_image_path):
        _imgs = [f for f in _imgs if not f[0] == '.']
        _dir[:] = [d for d in _dir if not d[0] == '.']
        for idx in range(len(_imgs)):
            _img = _imgs[idx]
            _img_path = os.path.join(_root, _img)
            pool.apply_async(blur_color_processing, (_root, _img_path, _img))
            #blur_color_processing(_root, _img_path, _img)
    pool.close()
    pool.join()
