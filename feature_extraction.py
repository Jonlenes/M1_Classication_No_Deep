from scipy.stats import kurtosis, skew
from joblib import Parallel, delayed
from dataset import CameraDataset
from math import ceil
from pandas import DataFrame as df
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from augmentation import Augmentator

import pywt
import numpy as np
import multiprocessing

import cv2

path_load_dataset = "/home/jonlenes/Desktop/dataset/"
path_save_csv = "/home/jonlenes/Desktop/dataset_csv/"


"""
Processa cada image do dataset, realizando a extração de features e gera um csv
A extração foi realizada de forma paralela utilizando todos os cores disponiveis
na CPU. A idea é resolver o problema sem usar deep learning.
"""


def process_image(d, i):
    print("Processing %03d" % i)
    image = d[i]
    return get_features(image)


def get_bgr(image):
    """
    Separa os canais de uma imagem carregada em formato bgr
    """
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


radius = 8
num_ptos = 3 * radius
lbp = LocalBinaryPatterns(num_ptos, radius)


def get_statistics(a):
    """
    Calcula as estatisticas de um vetor de entrada
    """
    a = a.flatten()
    return [np.mean(a), np.var(a), skew(a), kurtosis(a), np.std(a)]


def get_features(image):
    """
    Realiza a extração de feature para uma imagem de entrada:
        Histograma, Ruido, Wavelet, LBP... 
    """
    features = []

    denoised_img = cv2.fastNlMeansDenoisingColored(image)
    noise = image - denoised_img

    chs = get_bgr(noise)
    for ch in chs:
        hist = lbp.describe(ch)
        features.extend(hist.ravel())

    imgs = [image, noise]

    for idx in range(len(imgs)):
        img = imgs[idx]
        rgb = get_bgr(img)

        for channel in rgb:
            for i in range(3):
                cA, (coeffs) = pywt.dwt2(channel, 'db8')
                for coeff in coeffs:
                    features.extend(get_statistics(coeff))
                    hist = coeff.flatten()
                channel = cA

    return features


def make_datasets():
    """
    Ler os arquivos de entrada, faz a paralelização, faz o join das threads e salvo os features no .csv
    """

    feature_extract_id = 31
    names = ["train", "test"]

    dataset_train = CameraDataset(path_load_dataset + "train_files", path_load_dataset, Augmentator(), False, True)
    dataset_test = CameraDataset(path_load_dataset + "test_files", path_load_dataset, None)
    datasets = [dataset_train, dataset_test]

    num_cores = multiprocessing.cpu_count()
    num_imagens_by_time = 200
    print("Numeros de cores utilizando no processamento:", num_cores)

    y = datasets[0].get_labels()
    df(y).to_csv(path_save_csv + "{1}_target_{0}.csv".format(names[0], feature_extract_id), index=False)

    for d in range(0, 2):

        print("Processing {0} images: {1}".format(len(datasets[d]), names[d]))
        file_path = path_save_csv + "{1}_{0}.csv".format(names[d], feature_extract_id)

        for i in range(int(ceil(len(datasets[d]) / num_imagens_by_time))):
            # calc the block
            begin = i * num_imagens_by_time
            n = min(begin + num_imagens_by_time, len(datasets[d]))

            x = Parallel(n_jobs=num_cores)(delayed(process_image)(datasets[d], i) for i in range(begin, n))
            df(x).to_csv(file_path, mode='a', header=(i == 0), index=False)

    print(dataset_train.get_manip_count())


if __name__ == '__main__':
    make_datasets()
