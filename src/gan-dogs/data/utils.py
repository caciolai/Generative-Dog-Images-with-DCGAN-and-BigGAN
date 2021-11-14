import os
import xml.etree.ElementTree as ET


def load_bbox(file, breed_map, ann_dir):
    """Loads the bounding box for a given dog image file

    Args:
        file (str): path to dog image file
        breed_map (dict): map from dog image file to breed
        ann_dir (dict): path to directory containing bounding box information for each dog image file

    Returns:
        (int, int, int, int): bounding box
    """
    file = os.path.join(breed_map[file.split('_')[0]], file.split('.')[0])
    path = os.path.join(ann_dir, file)
    tree = ET.parse(path)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

    return (xmin, ymin, xmax, ymax)


def get_resized_bbox(height, width, bbox):
    """Make square bounding boxes of original ones, to keep a dog's aspect ratio.

    Args:
        height (int): 
        width (int): 
        bbox (int, int, int, int): bounding box

    Returns:
        (int, int, int, int): bounding box
    """

    lol = "Make square bounding boxes of original ones, to keep a dog's aspect ratio."
    xmin, ymin, xmax, ymax = bbox
    xlen = xmax - xmin
    ylen = ymax - ymin

    if xlen > ylen:
        diff = xlen - ylen
        min_pad = min(ymin, diff//2)
        max_pad = min(height-ymax, diff-min_pad)
        ymin = ymin - min_pad
        ymax = ymax + max_pad

    elif ylen > xlen:
        diff = ylen - xlen
        min_pad = min(xmin, diff//2)
        max_pad = min(width-xmax, diff-min_pad)
        xmin = xmin - min_pad
        xmax = xmax + max_pad

    return xmin, ymin, xmax, ymax
