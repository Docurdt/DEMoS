import staintools
from pathlib import Path
import re
import os
import cv2 as cv
import multiprocessing
import numpy as np

# Verifica se o OpenCV foi compilado com suporte a CUDA
if cv.cuda.getCudaEnabledDeviceCount() == 0:
    raise RuntimeError("OpenCV não foi compilado com suporte a CUDA. Instale o OpenCV com suporte a CUDA.")

def create_new_folder(_patient_path, ori_str, replace_str):
    _new_patient_path = re.sub(ori_str, replace_str, _patient_path)
    Path(_new_patient_path).mkdir(parents=True, exist_ok=True)
    return _new_patient_path

def variance_of_laplacian(image):
    # Converte a imagem para escala de cinza na GPU
    gpu_gray = cv.cuda.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Aplica o Laplacian na GPU
    gpu_laplacian = cv.cuda.Laplacian(gpu_gray, cv.CV_64F)
    # Baixa o resultado para a CPU e calcula a variância
    laplacian = gpu_laplacian.download()
    return laplacian.var()

def find_blur(imagePath):
    imagePath = Path(imagePath).resolve()
    # Carrega a imagem na CPU
    image = cv.imread(str(imagePath))

    if image is None:
        print(f"Failed to read image at: {imagePath}")
        return None, 0

    # Upload da imagem para a GPU
    gpu_image = cv.cuda_GpuMat()
    gpu_image.upload(image)

    # Calcula o desfoque na GPU
    fm = variance_of_laplacian(gpu_image)
    return gpu_image, fm

def blur_color_processing(_root, _img_path, _img, _img_path_template, _blur_threshold=1000):
    gpu_image, fm = find_blur(_img_path)
    if gpu_image is None:
        return fm  # Se a imagem não pôde ser lida, retorna diretamente

    print(f"Processing: {_img_path}, Blur score: {fm}")

    if fm <= _blur_threshold:
        # Não armazenar imagens desfocadas
        print(f"Image is blurred and will not be saved: {_img_path}")
    else:
        # Para imagens nítidas
        _new_img_folder = create_new_folder(_root, "tiling", "tiling_macenko")
        _new_img_path = os.path.join(_new_img_folder, _img)

        # Verifica se a imagem já existe no diretório de destino
        if os.path.exists(_new_img_path):
            print(f"Image already exists, skipping: {_new_img_path}")
            return fm

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

def pre_processing(extra_prefix="", _img_path_template=absolute_path):
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