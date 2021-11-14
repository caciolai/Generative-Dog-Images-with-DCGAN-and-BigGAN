import tensorflow as tf


@tf.function
def augment(img):
    """Augments the current image

    Args:
        img (tf.Tensor): image

    Returns:
        tf.Tensor: augmented image
    """
    # random mirroring
    img = tf.image.random_flip_left_right(img)

    # # randomly adjust saturation and brightness
    # img = tf.image.random_saturation(img, 0, 1)
    # img = tf.image.random_brightness(img, 0.1)
    # img = tf.clip_by_value(img, -1., 1.)

    return img


@tf.function
def preprocess(img, y):
    """Preprocessing pipeline for a (image, label) sample

    Args:
        img (tf.Tensor): image
        y (tf.Tensor): label

    Returns:
        (tf.Tensor, tf.Tensor): transformed (image, label) sample
    """
    # normalization in [-1, 1]
    # img = normalize(img)

    # data augmentation
    img = augment(img)

    return img, y
