#!/bin/bash

# Garantir que o Conda seja carregado corretamente
source ~/anaconda3/etc/profile.d/conda.sh  # Ou ~/miniconda3/etc/profile.d/conda.sh
conda create --name demos python=3.8 -y
conda activate demos

# Instalar pacotes
if [ -f ~/SageMaker/DEMoS/requirements.txt ]; then
    pip install -r ~/SageMaker/DEMoS/requirements.txt
else
    echo "Arquivo requirements.txt não encontrado!"
    exit 1
fi

# Instalar dependências do OpenSlide
sudo yum install -y openslide openslide-devel
conda install -c conda-forge openslide-python -y

# Criar diretório e montar EFS
sudo mkdir -p /mnt/efs-tcga
if ! command -v mount.efs &> /dev/null; then
    echo "amazon-efs-utils não encontrado. Instalando..."
    sudo yum install -y amazon-efs-utils
fi
sudo mount -t efs -o tls fs-078deca34f86658aa:/ /mnt/efs-tcga
sudo chmod 777 /mnt/efs-tcga/
