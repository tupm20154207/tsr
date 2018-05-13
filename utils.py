import numpy as np
import cv2
import random as rd
import tensorflow as tf
import matplotlib.pyplot as plt

N_CLASSES = 43
INP_WIDTH = 32
INP_HEIGHT = 32
TRAINING_DIR = './GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
TEST_DIR = './GTSRB_Test/Online-Test-sort/'

TRAIN_FEATURES = 'metadata/training_features'
TRAIN_LABELS = 'metadata/training_labels'
VAL_FEATURES = 'metadata/validation_features'
VAL_LABELS = 'metadata/validation_labels'
TEST_FEATURES = 'metadata/test_features'
TEST_LABELS = 'metadata/test_labels'


def parse_data(issave=False):

    training_features = []
    training_labels = []

    validation_features = []
    validation_labels = []

    test_features = []
    test_labels = []

    for i in range(N_CLASSES):

        offs = str(i).zfill(5)
        training_class_dir = TRAINING_DIR + offs + '/'
        training_file_name = training_class_dir + 'GT-' + offs + '.csv'

        test_class_dir = TEST_DIR + offs + '/'
        test_file_name = test_class_dir + 'GT-' + offs + '.csv'

        with open(training_file_name, 'r') as f:
            annotations = f.readlines()[1:]

            val_offsets = rd.sample(range(int(annotations[-1][:5])+1), 5)

            for line in annotations:
                im_info = line.split(';')
                track_number = int(line[:5])

                # Load image
                img = cv2.imread(
                    training_class_dir + im_info[0],
                    cv2.IMREAD_GRAYSCALE
                )

                # Crop the sub-rectangle associate with the sign position
                sign_img = img[int(im_info[4]):int(im_info[6]),
                               int(im_info[3]):int(im_info[5])]

                # Resize image
                fixed_size_arr = cv2.resize(sign_img, (INP_WIDTH, INP_HEIGHT))

                # # Apply histogram equalization:
                # equalized = cv2.equalizeHist(fixed_size_arr)

                # Rescale image to range (0,1)
                img_norm = cv2.normalize(fixed_size_arr.astype(
                    np.float32), None, 0, 1, cv2.NORM_MINMAX)

                if track_number in val_offsets:
                    validation_features.append(img_norm)
                    validation_labels.append(np.int32(im_info[7]))
                else:
                    training_features.append(img_norm)
                    training_labels.append(np.int32(im_info[7]))

        with open(test_file_name, 'r') as f1:

            annotations = f1.readlines()[1:]

            for line in annotations:
                im_info = line.split(';')

                # Load image
                img = cv2.imread(test_class_dir +
                                 im_info[0], cv2.IMREAD_GRAYSCALE)

                # Crop the sub-rectangle associate with the sign position
                sign_img = img[int(im_info[4]):int(
                    im_info[6]), int(im_info[3]):int(im_info[5])]

                # Resize image
                fixed_size_arr = cv2.resize(sign_img, (INP_WIDTH, INP_HEIGHT))

                # # Apply histogram equalization:
                # equalized = cv2.equalizeHist(fixed_size_arr)

                # Rescale image to range (0,1)
                img_norm = cv2.normalize(fixed_size_arr.astype(
                    np.float32), None, 0, 1, cv2.NORM_MINMAX)

                test_features.append(img_norm)
                test_labels.append(np.int32(im_info[7]))

    training_features = np.asarray(training_features)
    training_labels = get_one_hot(np.asarray(training_labels))
    validation_features = np.asarray(validation_features)
    validation_labels = get_one_hot(np.asarray(validation_labels))
    test_features = np.asarray(test_features)
    test_labels = get_one_hot(np.asarray(test_labels))

    # Write result to files
    if issave:
        np.save(TRAIN_FEATURES, training_features)
        np.save(TRAIN_LABELS, training_labels)
        np.save(VAL_FEATURES, validation_features)
        np.save(VAL_LABELS, validation_labels)
        np.save(TEST_FEATURES, test_features)
        np.save(TEST_LABELS, test_labels)

    return (training_features, training_labels), \
           (validation_features, validation_labels), \
           (test_features, test_labels)


def load_data(flush=False):

    if flush:
        return parse_data(True)

    return (np.load(TRAIN_FEATURES+'.npy'), np.load(TRAIN_LABELS+'.npy')), \
           (np.load(VAL_FEATURES+'.npy'), np.load(VAL_LABELS+'.npy')), \
           (np.load(TEST_FEATURES+'.npy'), np.load(TEST_LABELS+'.npy'))


def get_tf_dataset(batch_size, flush=False):
    train, val, test = load_data(flush)

    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.batch(batch_size)

    validate_data = tf.data.Dataset.from_tensor_slices(val)
    validate_data = validate_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, validate_data, test_data


def get_one_hot(labels):
    n_samples = np.size(labels, axis=0)
    one_hot_labels = np.zeros((n_samples, N_CLASSES))
    one_hot_labels[np.arange(n_samples), labels] = 1

    return one_hot_labels


def read_training_info(path):
    with open(path, 'r') as f:
        epoch = []
        loss = []
        accuracy = []
        time = []
        for line in f.readlines():
            info = line[:-1].split(';')
            epoch.append(int(info[0]))
            loss.append(float(info[1]))
            accuracy.append(float(info[2]))
            time.append(float(info[3]))
        return epoch, loss, accuracy, time


def plot():
    epoch, loss, acc, _ = read_training_info('training_info/le_net_ss/info')
    e1, l1, a1, _ = read_training_info('training_info/le_net_ms/info')
    plt.plot(epoch, loss, 'b')
    plt.plot(epoch, acc, 'r')
    plt.plot(e1, l1, '--b')
    plt.plot(e1, a1, '--r')
    plt.show()


if __name__ == '__main__':
    # plot()
    train, val, test = load_data(False)
    print(np.shape(train[0]), np.shape(val[0]), np.shape(test[0]))
    print(np.shape(train[1]), np.shape(val[1]), np.shape(test[1]))
