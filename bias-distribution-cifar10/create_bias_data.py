import numpy as np
import matplotlib.pyplot as plt
import pickle

# https://github.com/snatch59/load-cifar-10/blob/master/load_cifar_10.py

"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


if __name__ == "__main__":
    cifar_10_dir = 'cifar-10-batches-py'

    test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    print("original distribution:")
    unique, counts = np.unique(test_labels, return_counts=True)
    print(np.asarray((unique, counts)).T)

    bias_distribution = [1000,10,20,400,200,1000,100,200,1,2]
    count = np.zeros(10)

    new_test_data = []
    new_test_filenames = []
    new_test_labels = []
    new_label_names = []

    for i in range(0,10000):
        if count[test_labels[i]] >= bias_distribution[test_labels[i]]:
            continue

        new_test_data.append(test_data[i])
        new_test_filenames.append(test_filenames[i])
        new_test_labels.append(test_labels[i])
        count[test_labels[i]] += 1

    print("Test data: ", len(new_test_data))
    print("Test filenames: ", len(new_test_filenames))
    print("Test labels: ", len(new_test_labels))
