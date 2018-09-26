import cv2
import numpy as np

from pre_process_dataset import path_load_dataset
from dataset import CameraDataset
from data_augmentation import Augmentator


files = CameraDataset(path_load_dataset + "train_files", path_load_dataset, None).get_files_name()
sz = 512


def cut_image_center(image):
    """
    Corta uma imagem no centro
    """
    h, w = image.shape[0:2]
    return image[int((h - sz) / 2):int((h + sz) / 2), int((w - sz) / 2):int((w + sz) / 2)]


def cut_all_images():
    """
    Corta todas as imagns do dataset
    """
    for i, file in enumerate(files):
        print("Processing %03d" % i)
        im = cv2.imread(file)
        im = cut_image_center(im)
        cv2.imwrite(file, im)


def expand_dataset():
    """
    Realiza o data augmentation no dataset, gerando um dataset maior
    """
    
    augmentator = Augmentator()

    for i, file in enumerate(files):
        print("Processing %03d" % i)
        im = cv2.imread(file)
        file = file.split(".")
        for j in range(11):
            im_copy = np.copy(im)
            im_copy = augmentator(im_copy, j)
            cv2.imwrite(file[0] + "_" + str(j + 1) + "." + file[1], im_copy)

