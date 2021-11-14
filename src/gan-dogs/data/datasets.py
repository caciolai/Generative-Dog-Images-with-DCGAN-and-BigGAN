from pathlib import Path
import os
import cv2
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
import albumentations as A
import tensorflow as tf

from .utils import load_bbox, get_resized_bbox
from .preprocessing import preprocess


DATA_DIR = Path("../../data")
IMAGES_DIR = DATA_DIR / "Images"
ANN_DIR = DATA_DIR / "Annotation"


def download_dataset():
    """
    Downloads the StanfordDogs dataset.
    """
    os.system(
        "wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -P ./data")
    os.system(
        "wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar -P ./data")
    os.system("tar xf ./data/images.tar -C ./data")
    os.system("tar xf ./data/annotation.tar -C ./data")


def prepare_raw_dataset():
    """Prepares the raw dataset from the StanfordDogs dataset present on disk.

    Returns:
        (np.array, np.array): (training images, training labels)
    """
    all_breeds = os.listdir(IMAGES_DIR)
    all_files = [file for breed in all_breeds for file in os.listdir(
        os.path.join(IMAGES_DIR, breed))]

    breeds = glob.glob(ANN_DIR+'*')
    annotations = []
    for breed in breeds:
        annotations += glob.glob(breed+'/*')

    breed_map = {}
    for annotation in annotations:
        breed = annotation.split('/')[-2]
        index = breed.split('-')[0]
        breed_map.setdefault(index, breed)

    all_labels = [breed_map[file.split('_')[0]] for file in all_files]

    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels)
    all_labels = all_labels.astype(np.int32)

    all_bboxes = [load_bbox(file) for file in all_files]

    print('Total files       : {}'.format(len(all_files)))
    print('Total labels      : {}'.format(len(all_labels)))
    print('Total bboxes      : {}'.format(len(all_bboxes)))
    print('Total annotations : {}'.format(len(annotations)))
    print('Total classes     : {}'.format(len(le.classes_)))

    resized_bboxes = []
    for file, bbox in zip(all_files, all_bboxes):
        file = os.path.join(breed_map[file.split('_')[0]], str(file))
        path = os.path.join(IMAGES_DIR, file)
        img = Image.open(path)
        width, height = img.size
        xmin, ymin, xmax, ymax = get_resized_bbox(height, width, bbox)
        resized_bboxes.append((xmin, ymin, xmax, ymax))

    all_images = []
    dim = 64
    for file, bbox in tqdm(zip(all_files, resized_bboxes), total=len(all_files)):
        file = os.path.join(breed_map[file.split('_')[0]], str(file))
        path = os.path.join(IMAGES_DIR, file)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = bbox
        img = img[ymin:ymax, xmin:xmax]

        transform = A.Compose([A.Resize(dim, dim, interpolation=cv2.INTER_AREA),
                               A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform(image=img)['image']
        all_images.append(img)

    all_images = np.array(all_images)

    return all_images, all_labels


def prepare_dataset(train_data, train_labels, batch_size):
    """Prepares the tensorflow training dataset.

    Args:
        train_data (np.array): Training images.
        train_labels (np.array): Training labels (breeds).

    Returns:
        tf.data.Dataset: Tensorflow training dataset.
    """

    # Build Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data, train_labels))
    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess)

    # Shuffle and batch
    train_dataset = train_dataset.shuffle(
        buffer_size=train_data.shape[0]).batch(batch_size)

    # For performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset
