import glob
import math

import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm

'''
This file contains the data processing function for the ResNet model

Functions:
# load data from the input file
def load_data(path, test): path--the full path to the folder which input image is stored starting from the root dirct
                           test--label to convert the training set and testing set, default=false

# sample visualization for the input image, will randomly select 12 images
visualize_image(image_data): image_data--set of image to visualize
'''


def load_dataset(train_label=True):
    # list to contain the loaded tuples
    data_tuple = []
    CT_labels = ['/Covid', '/Normal', '/Viral Pneumonia']
    if train_label:
        # load all the images
        for label in CT_labels:
            path = 'Covid19-dataset/train'
            path += label + '/*.png'
            for image in glob.glob(path):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                # Remove the '/' character from 'label' iterator
                data_tuple.append((img, label[1:]))
            path = 'Covid19-dataset/train'
            path += label + '/*.jpeg'
            for image in glob.glob(path):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                # Remove the '/' character from 'label' iterator
                data_tuple.append((img, label[1:]))
            path = 'Covid19-dataset/train'
            path += label + '/*.jpg'
            for image in glob.glob(path):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                # Remove the '/' character from 'label' iterator
                data_tuple.append((img, label[1:]))
        np.random.shuffle(data_tuple)

    else:
        for label in CT_labels:
            path = 'Covid19-dataset/test'
            path += label + '/*.png'
            for image in glob.glob(path):
                img = cv2.imread(image, cv2.COLOR_RGB2BGR)
                # Remove the '/' character from 'label' iterator
                data_tuple.append((img, label[1:]))
            path = 'Covid19-dataset/test'
            path += label + '/*.jpeg'
            for image in glob.glob(path):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                # Remove the '/' character from 'label' iterator
                data_tuple.append((img, label[1:]))
            path = 'Covid19-dataset/test'
            path += label + '/*.jpg'
            for image in glob.glob(path):
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                # Remove the '/' character from 'label' iterator
                data_tuple.append((img, label[1:]))
        np.random.shuffle(data_tuple)
    return data_tuple


def load_data(path, test=False):
    if test:
        transformer = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomGrayscale(p=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(path, transform=transformer)
        return dataset

    transformer = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(path, transform=transformer)
    return dataset


def visualize_image(image_data):
    num_view = 12
    image_view = []
    tag_view = []

    for i in range(num_view):
        index = np.random.randint(low=0, high=len(image_data) - 1)
        image, tag = image_data[index]
        image = image.detach().numpy().transpose(1, 2, 0)
        image_view.append(image)
        tag_view.append(tag)

    plt.figure(figsize=(15, 10))
    for img in tqdm(range(num_view)):
        ax = plt.subplot(math.ceil(num_view / 4), 4, img + 1)
        plt.imshow(image_view[img], cmap=plt.cm.bone)
        plt.title("Covid" if tag_view[img] == 0 else "Normal" if tag_view[img] == 1 else "Viral Pneumonia")
        plt.axis("off")
    plt.savefig('./output/image_view.png')
    plt.show()


if __name__ == '__main__':
    dataset = load_data('Covid19-dataset/train')
    print(dataset.class_to_idx)
    print(dataset)
    visualize_image(dataset)
