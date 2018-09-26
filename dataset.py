import os
import cv2

from post_processing import TTA_COUNT

class CameraDataset:
    """
    Classe que abstrai todo o dataset
    """

    def __init__(self, path_list, path_image, augmentator, expand_dataset=False, use_probability=False):
        self._files = []
        self._labels = []

        with open(path_list) as f:
            for example in f.read().split("\n")[:-1]:
                if " " in example:
                    example = example.split(" ")
                    file_path = path_image + " ".join(example[:-1])
                    label = int(example[-1])
                else:
                    file_path, label = path_image + example, -1

                suffixes = ["JPG", "JPEG", "jpeg"]
                if not os.path.exists(file_path):
                    for suffix in suffixes:
                        pos = file_path.rfind(".")
                        file_path = file_path[:pos]
                        file_path += "." + suffix

                        if os.path.exists(file_path):
                            break

                self._files.append(file_path)
                self._labels.append(label)

        self._augmentator = augmentator
        self._use_probability = use_probability

    def _get_patch(self, index):
        image = cv2.imread(self._files[index])

        if self._use_probability:
            return self._augmentator(image)

        return image

    def get_labels(self):
        return self._labels

    def get_files_name(self):
        return self._files

    def get_manip_count(self):
        return self._augmentator.get_manip_count()

    def __getitem__(self, index):
        image = self._get_patch(index)
        return image

    def __len__(self):
        return len(self._files)
