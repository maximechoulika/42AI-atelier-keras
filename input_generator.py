import cv2
import random
from random import randint

import numpy as np

from helper import load_image

class InputGenerator(object):

    def __init__(self, dataset, nb_classes):
        self.dataset    = dataset
        self.nb_classes = nb_classes

    def augGrayScale(self, input):
        input = np.dot(input[...,:3], [0.299, 0.587, 0.114])
        input = np.reshape(input, (input.shape[0], input.shape[1], 1))
        input = np.concatenate((input, input, input), axis=2)
        return input

    def augCoarseDropout(self, input):
        scale, shape = random.randint(5, 30), input.shape[:2]
        shape = (np.int32(shape[0] / scale), np.int32(shape[1] / scale))
        mask = np.random.randint(2, size=shape).reshape([shape[0], shape[1], 1]) | np.random.randint(2, size=shape).reshape([shape[0], shape[1], 1])
        mask = np.concatenate((mask, mask, mask), axis=2)
        input = input * cv2.resize(np.uint8(mask), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
        return input

    def input_generator(self, batchsize=32):
        buff_train = self.dataset.copy()

        while True:
            if batchsize > buff_train.shape[0]:
                buff_train = self.dataset.copy()

            X = []
            y = []
            for _ in range(batchsize):
                idx = randint(0, buff_train.shape[0] - 1)
                class_id, img_path = buff_train[idx]

                buff_train = np.delete(buff_train, [idx], 0)

                img = load_image(img_path)

                # Image data_aumentation
                if random.random() > 0.25:
                    img = self.augGrayScale(img)
                if random.random() > 0.25:
                    img = self.augCoarseDropout(img)
                # Data normalization
                img = img / 255.

                # Onehot encoding
                id = np.zeros(self.nb_classes)
                id[int(class_id)] = 1

                X.append(img)
                y.append(id)

            X = np.array(X)
            y = np.array(y)
            yield X, y
