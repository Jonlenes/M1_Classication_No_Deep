from collections import Counter
import numpy as np
import copy

TTA_COUNT = 11

CLASSES = [
    "HTC-1-M7",
    "LG-Nexus-5x",
    "Motorola-Droid-Maxx",
    "Motorola-Nexus-6",
    "Motorola-X",
    "Samsung-Galaxy-Note3",
    "Samsung-Galaxy-S4",
    "Sony-NEX-7",
    "iPhone-4s",
    "iPhone-6"
]

def generate_submit_simple(y_pred, files, output_file):
    """
    Gera o cvs para a submissão no Kaggle
    """
    labels = list(map(lambda y: CLASSES[y], y_pred))
    filenames = list(map(lambda x: x[x.rfind("/") + 1:], files))

    with open(output_file, "w") as f:
        f.write("fname,camera\n")
        for i in range(len(y_pred)):
            f.write(filenames[i])
            f.write(",")
            f.write(labels[i])
            f.write("\n")