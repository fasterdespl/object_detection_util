""" This file is use to update tensorflow. The code will be hidden here in a function instead being in the notebook
"""
import os
import sys
ROOT_DIR = os.path.abspath("/content/maskrcnn_colab")
# Import Mask RCNN
sys.path.append(ROOT_DIR)


print("Updating Tensorflow to version 2.5.0")
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
!dpkg -i libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb
# Check if package has been installed
!ls -l /usr/lib/x86_64-linux-gnu/libcudnn.so.*
# Upgrade Tensorflow
!pip install --upgrade tensorflow==2.5.0