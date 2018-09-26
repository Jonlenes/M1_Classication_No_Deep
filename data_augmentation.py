import random
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
from torchvision import transforms


CROP_SIZE = 224
MANIP_PROBABILITY = 0.5


class Augmentator(object):
    """
    Implementa Data Augmentator - Baseado em outra classe da Internet
    """

    def __init__(self):

        self._manip_list = []

        self._manip_list.append([Augmentator._jpg_manip, 70])
        self._manip_list.append([Augmentator._jpg_manip, 90])

        self._manip_list.append([Augmentator._gamma_manip, 0.8])
        self._manip_list.append([Augmentator._gamma_manip, 1.2])

        self._manip_list.append([Augmentator._bicubic_manip, 0.5])
        self._manip_list.append([Augmentator._bicubic_manip, 0.8])
        self._manip_list.append([Augmentator._bicubic_manip, 1.5])
        self._manip_list.append([Augmentator._bicubic_manip, 2.0])

        self._manip_count = 0


    @staticmethod
    def _get_transforms():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    @staticmethod
    def _jpg_manip(image, quality):
        buffer = BytesIO()
        image = Image.fromarray(image)
        image.save(buffer, format='jpeg', quality=quality)
        buffer.seek(0)
        result = Image.open(buffer)
        result = np.array(result)
        return result


    @staticmethod
    def _gamma_manip(image, gamma):
        result = np.uint8(cv2.pow(image / 255., gamma) * 255.)
        return result


    @staticmethod
    def _bicubic_manip(image, scale):
        result = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return result


    def __call__(self, image):
        if random.random() < MANIP_PROBABILITY:
            manip_func, param = random.choice(self._manip_list)
            image = manip_func(image, param)
            self._manip_count += 1

        return image


    def get_manip_count(self):
        return self._manip_count
