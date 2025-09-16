import staintools
from pathlib import Path
import re
import os
import cv2 as cv
from shutil import copy2  # Usar copy2 para manter metadados
import multiprocessing

'''
Function description:
    Detect the blurred images and move them to a backup directory;
    Correct all the clear images into a unified color space;
'''

def create_new_folder(_patient_path, ori_str, replace_str):
    _new_patient_path = re.sub(ori_str, replace_str, _patient_path)
    Path(_new_patient_path).mkdir(parents=True, exist_ok=True)
    return _new_patient_path


def variance_of_laplacian(image):
    return cv.Laplacian(image, cv.CV_64F).var()


def find_blur(imagePath):
    imagePath = Path(imagePath).resolve()
    image = cv.imread(str(imagePath))

    if image is None:
        print(f"Failed to read image at: {imagePath}")
        return None, 0

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return image, fm



def blur_color_processing(_root, _img_path, _img, _img_path_template, _blur_threshold=1000):
    img, fm = find_blur(_img_path)
    if img is None:
        return fm  # Se a imagem não pôde ser lida, retorna diretamente

    print(f"Processing: {_img_path}, Blur score: {fm}")

    if fm <= _blur_threshold:
        # Para imagens desfocadas
        blur_path = create_new_folder(_root, "tiling", "tiling_blur")
        new_img_path = os.path.join(blur_path, _img)
        success = copy(_img_path, new_img_path)
        if success:
            print(f"Blurred image moved to: {new_img_path}")
        else:
            print(f"Failed to move blurred image to: {new_img_path}")
    else:
        # Para imagens nítidas
        _new_img_folder = create_new_folder(_root, "tiling", "tiling_macenko")
        _new_img_path = os.path.join(_new_img_folder, _img)

        try:
            # Processo de normalização de cor usando Macenko
            template_image = Path(_img_path_template)
            template_image = str(template_image.resolve())
            target = staintools.read_image(template_image)

            if target is None:
                raise ValueError(f"Template image not found or failed to load: {template_image}")

            normalizer = staintools.StainNormalizer(method='macenko')
            normalizer.fit(target)

            # Ler a imagem de entrada
            image = staintools.read_image(_img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {_img_path}")
            
            image = staintools.LuminosityStandardizer.standardize(image)
            img = normalizer.transform(image)

            # Salvar a imagem no diretório de destino
            success = cv.imwrite(_new_img_path, img)
            if success:
                print(f"Image transformed and saved to: {_new_img_path}")
            else:
                raise IOError(f"Failed to save image at: {_new_img_path}")

        except Exception as e:
            print(f"Error during image processing: {e}")

    return fm




# Caminho relativo ao diretório do projeto
relative_path = "/mnt/efs-tcga/DEMoS/HEAL/Pre_processing/img_template.png"
project_base = os.getcwd()
# Construir o caminho absoluto
absolute_path = os.path.join(project_base, relative_path)


def pre_processing(extra_prefix="", _img_path_template = absolute_path):
    print("[INFO] Starting blur detection ...")
    cpu_num = multiprocessing.cpu_count()
    print(f"The CPU number of this machine is {cpu_num}")
    pool = multiprocessing.Pool(cpu_num)

    _image_path = "/mnt/efs-tcga/HEAL_Workspace/tiling" + str(extra_prefix)
    for _root, _dir, _imgs in os.walk(_image_path):
        _imgs = [f for f in _imgs if not f[0] == '.']
        _dir[:] = [d for d in _dir if not d[0] == '.']

        for idx in range(len(_imgs)):
            _img = _imgs[idx]
            _img_path = os.path.join(_root, _img)

            # Processamento paralelo com multiprocessing
            pool.apply_async(blur_color_processing, (_root, _img_path, _img, _img_path_template))

    pool.close()
    pool.join()
