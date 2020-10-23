import numpy as np
import cv2 as cv
import os

from PIL import Image


class Dataset:
    def __init__(self, root):
        self.root = root
        self.images = []
        self.labels = []

    def load_numpy_data(self, augment=False):
        image_names = os.listdir(self.root)

        for i in range(len(image_names)):
            image = Image.open(os.path.join(self.root, image_names[i])).convert('L')
            image = np.array(image)
            image = Dataset.reverse_color(image)
            if augment:
                images_aug = self.augment_image(image)
                for i in range(len(images_aug)):
                    if len(image.shape) == 2:
                        images_aug[i] = np.expand_dims(images_aug[i], axis=0)
                    images_aug[i] = images_aug[i] / 255.
                self.images += images_aug if augment else image
            else:
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
                image = image / 255.
                self.images.append(image)

            label, _, _ = os.path.basename(image_names[i]).split('.')
            self.labels += [label] * len(images_aug) if augment else [label]

        self.labels = Dataset.to_categorical(self.labels, 10)

        print('{} images loaded'.format(len(self.images)))
        print('{} labels loaded'.format(len(self.labels)))

    def load_image_data(self):
        image_names = os.listdir(self.root)

        for i in range(len(image_names)):
            image = Image.open(os.path.join(self.root, image_names[i])).convert('L')
            self.images.append(image)

            label, _, _ = os.path.basename(image_names[i]).split('.')
            self.labels.append(label)

        self.labels = Dataset.to_categorical(self.labels, 10)

    def augment_image(self, img, rotate=True):
        imgs_aug = []
        if rotate:
            rotate_func = Dataset.rotate_image
            for d in range(-30, 31, 10):
                imgs_aug.append(rotate_func(img, d, 1))

        return imgs_aug


    @staticmethod
    def rotate_image(img, degree, scale):
        h, w = img.shape
        M = cv.getRotationMatrix2D((h / 2, w / 2), degree, scale)
        img_rot = cv.warpAffine(img, M, (h, w))

        return img_rot

    @staticmethod
    def reverse_color(image):
        return 255 - image

    @staticmethod
    def to_categorical(labels, n_class):
        one_hots = []
        for label in labels:
            one_hot = [0 for i in range(n_class)]
            one_hot[int(label)] = 1
            one_hots.append(one_hot)

        return np.array(one_hots)





