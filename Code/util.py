import numpy as np
import struct
import matplotlib.pyplot as plt
from array import array
import random

def read_images_labels(images_path, labels_path):
    # read images
    # rb: read binary mode
    with open(images_path, 'rb') as file:
        # deal with header: 32 bit integer in the first 4 rows
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', file.read(16))
        # B: unsigned char
        fmt_image = array('B', file.read())
        images = np.empty((num_images, num_rows, num_cols))
        # read one image at a time sequentially
        for i in range(num_images):
            images[i] = np.array(fmt_image[i * num_rows * num_cols : (i+1) * num_rows * num_cols]).reshape(num_rows, num_cols)

    # read labels
    with open(labels_path, 'rb') as file:
        # deal with header: 32 bit integer in the first 2 rows
        magic_number, num_items = struct.unpack(">II", file.read(8))
        # B: unsigned char
        labels = array('B', file.read()).tolist()
        labels = np.asarray(labels)

    return images, labels


def load_dataset(training_mode = True):
    train_images_path = 'train-images-idx3-ubyte'
    train_labels_path = 'train-labels-idx1-ubyte'
    test_images_path = 't10k-images-idx3-ubyte'
    test_labels_path = 't10k-labels-idx1-ubyte'
    x_train, y_train = read_images_labels(train_images_path, train_labels_path)
    x_test, y_test = read_images_labels(test_images_path, test_labels_path)
    if training_mode:
        return x_train, y_train
    else:
        return x_test, y_test


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.savefig('mnist.png')
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1

# Show some random training and test images
x_train, y_train = load_dataset(training_mode=True)

x_test, y_test = load_dataset(training_mode=False)

images_2_show = []
titles_2_show = []

for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 6):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

# show_images(images_2_show, titles_2_show)
