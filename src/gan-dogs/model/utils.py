import tensorflow as tf
import numpy as np
import scipy
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import os
import cv2
from keras.applications.inception_v3 import InceptionV3

from ..data.configuration import get_configuration

CONFIG = get_configuration()


@tf.function
def smooth_real_labels(y):
    """
    Label smoothing -- technique from GAN hacks, instead of assigning 1/0 as class labels, 
    we assign a random number in range [0.7, 1.0] as label for real images
    """
    return y - tf.random.uniform(tf.shape(y), minval=0, maxval=CONFIG["label_smoothness"])


@tf.function
def smooth_fake_labels(y):
    """
    Label smoothing -- technique from GAN hacks, instead of assigning 1/0 as class labels, 
    we assign a random number in range [0.0, 0.3] as label for fake images
    """
    return y + tf.random.uniform(tf.shape(y), minval=0, maxval=CONFIG["label_smoothness"])


@tf.function
def noisy_labels(labels, p_flip):
    """Randomly flip some labels.

    Args:
        labels (tf.Tensor): labels to be flipped
        p_flip (float): probability of flipping a label

    Returns:
        tf.Tensor: noisy labels
    """

    # mask to select the labels to be flipped
    mask = tf.cast(
        (tf.random.uniform(tf.shape(labels), minval=0, maxval=1) < p_flip),
        dtype=tf.float32
    )
    flipped_labels = tf.subtract(1., labels)

    return mask * flipped_labels + (1. - mask) * labels

# scale an array of images to a new size


def scale_images(images, new_shape):
    """Scale images

    Args:
        images (np.array): images to scale
        new_shape (Tuple[int, int]): output image scale

    Returns:
        np.array: scaled images
    """
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_fid(images1, images2):
    """Calculate Frechet Inception Distance (FID) score of a GAN model.

    Args:
        images1 (np.array): target images (usually real images)
        images2 (np.array): other images (usually generated images)

    Returns:
        float: FID score
    """

    # prepare the inception v3 model
    inception_model = InceptionV3(
        include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # calculate activations
    act1 = inception_model.predict(images1)
    act2 = inception_model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def create_batch_images(folder, filename):
    """Create a batch of closely packed generated images from disk (to later create GIF showing training progress)

    Args:
        folder (str): folder with source images
        filename (str): path of resulting packed image
    """
    start_idx = 16

    files = [f for f in os.listdir(folder)]

    imgs = np.zeros([64*4, 64*4, 3])

    i, j = 0, 0
    for f in files:
        if 'Batch' in f:
            continue

        fname = f.split('.')[0]
        idx = int(fname.split('_')[1])
        if idx < start_idx:
            continue

        fpath = os.path.join(folder, f)
        img = cv2.imread(fpath)
        img = (1.0 / 255) * img

        imgs[i*64: (i+1)*64, j*64: (j+1)*64, :] = img
        j += 1
        if j == 4:
            i += 1
            j = 0

    plt.imshow(imgs)
    plt.show()

    fpath = os.path.join(folder, filename)
    imgs = (255 * imgs).astype('uint8')

    cv2.imwrite(fpath, imgs)
