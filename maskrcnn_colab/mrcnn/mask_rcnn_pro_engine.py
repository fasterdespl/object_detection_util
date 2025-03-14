import os
import sys
import random
import math
import re
import time


import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import os
import shutil
import zipfile
import imgaug
import glob

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/maskrcnn_colab")
print("VERS 0.9 - 28/12/2023")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from PIL import Image, ImageDraw



DRIVE_ROOT_DIR = "/content/gdrive/MyDrive/mrcnn/"




# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CustomConfig(Config):
    def __init__(self, num_classes, steps_epoch=500, image_size=832):
        self.NUM_CLASSES = num_classes + 1
        self.STEPS_PER_EPOCH = steps_epoch
        self.IMAGE_MAX_DIM = image_size
        self.IMAGE_MIN_DIM = image_size
        super().__init__()
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 832
    #IMAGE_MAX_DIM = 832

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = STEPS_PER_EPOCH // 100

    DETECTION_MIN_CONFIDENCE = 0.9



"""
NOTEBOOK PREFERENCES
"""
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class CustomDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_custom(self, annotation_json, images_dir, dataset_type="train"):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        #print("Annotation json path: ", annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()


        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']

            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}

        # Split the dataset, if train, get 90%, else 10%
        len_images = len(coco_json['images'])
        if dataset_type == "train":
            img_range = [int(len_images / 9), len_images]
        else:
            img_range = [0, int(len_images / 9)]

        for i in range(img_range[0], img_range[1]):
            image = coco_json['images'][i]
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        #print("Class_ids, ", class_ids)
        return mask, class_ids

    def count_classes(self):
        class_ids = set()
        n_images = 0
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            annotations = image_info['annotations']
            n_images += 1
            for annotation in annotations:
                class_id = annotation['category_id']
                class_ids.add(class_id)

        class_number = len(class_ids)
        return class_number, n_images

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "trained_models")

def load_training_model(config, project_name, init_with="coco"):
    model_dir = os.path.join(DRIVE_ROOT_DIR, project_name, "trained_models")
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_dir)

    # Which weights to start with?
    # = "coco"  # imagenet, coco, or last
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    return model


def display_image_samples(dataset_train):
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)

    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

def load_image_dataset(project_name):
    project_dir = os.path.join(DRIVE_ROOT_DIR, project_name)
    dataset_train = CustomDataset()
    dataset_val = CustomDataset()
    print("Annotation path", project_dir)
    # Find all Json annotations
    annotation_path = os.path.join(project_dir, "*.json")
    annotations_files_path = glob.glob(annotation_path)

    print("Found {} annotation files".format(len(annotations_files_path)))
    for file_path in annotations_files_path:
        dataset_train.load_custom(file_path, "/content/dataset", dataset_type='train')
        dataset_val.load_custom(file_path, "/content/dataset", dataset_type='val')

    dataset_train.prepare()
    dataset_val.prepare()

    class_number, n_images = dataset_train.count_classes()
    print("{} train images".format(n_images))
    _, n_images = dataset_val.count_classes()
    print("{} validation images".format(n_images))
    print("{} classes".format(class_number))

    return dataset_train, dataset_val, class_number


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

imgaug_aug = imgaug.augmenters.Sometimes(5/6,imgaug.augmenters.OneOf(
                                            [
                                            imgaug.augmenters.Fliplr(1),
                                            imgaug.augmenters.Flipud(1),
                                            imgaug.augmenters.Affine(rotate=(-45, 45)),
                                            imgaug.augmenters.Affine(rotate=(-90, 90)),
                                            imgaug.augmenters.Affine(scale=(0.5, 1.5))
                                             ]
                                        )
                                   )


def train_head(model, dataset_train, dataset_val, config, epochs, enable_aug=False):
    global imgaug_aug
    augmentation = imgaug_aug if enable_aug is True else None

    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs,
            layers='heads',
            augmentation=augmentation)


def train_all_layers(model, dataset_train, dataset_val, config, epochs, enable_aug=False):
    global imgaug_aug
    augmentation = imgaug_aug if enable_aug is True else None
    print("Augmentation")
    print(augmentation)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epochs,
                layers="all",
                augmentation=augmentation)


""" DETECTION TEST YOUR MODEL """

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def extract_images(my_zip, output_dir):
    # Make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(my_zip) as zip_file:
        count = 0
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue
            count += 1
            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(os.path.join(output_dir, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)
        print("Extracted: {} images".format(count))


def load_test_model(num_classes, project_name):
    inference_config = InferenceConfig(num_classes)
    model_dir = os.path.join(DRIVE_ROOT_DIR,  project_name, "trained_models")
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model, inference_config

def test_random_image(test_model, dataset_val, inference_config):
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # Model result
    print("Trained model result")
    results = test_model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax(), show_bbox=False, figsize=(12, 12),
                                title="inference_test")

    print("Annotation")
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_val.class_names, figsize=(12, 12),
                                title="annotated_image")



# Connect google drive
def connect_google_drive(project_name):
    from google.colab import drive
    drive.mount('/content/gdrive')

    model_dir = os.path.join(DRIVE_ROOT_DIR, project_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("New project created {}".format(project_name))
        print("You'll find the project on Google Drive, on the folder pysource_mrcnn_pro/{} .".format(project_name))
    else:
        print("Project {} already exists. Editing existing project.".format(project_name))
    return model_dir

def create_mrcnn_output_directory(project_name):
    model_dir = os.path.join(DRIVE_ROOT_DIR,  project_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("New project created {}".format(project_name))
        print("You'll find the project on Google Drive, pysource_mrcnn_pro/{} .".format(project_name))
    else:
        print("Project {} already exists. Editing existing project.".format(project_name))
    return model_dir


def model_evaluation(dataset_val, test_model, inference_config):
    APs = []
    print("Testing the model on {} validation images.".format(len(dataset_val.image_ids)))
    for image_id in dataset_val.image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = test_model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    return APs


def path_to_tb_path(project_name):
    """ This functions adds \\ to the path otherwise tensorboard doesn't recognize it"""
    logs_path = os.path.join(DRIVE_ROOT_DIR, project_name, "trained_models")
    path = os.path.normpath(logs_path)
    path_split = path.split(os.sep)

    final_path = ""
    for folder in path_split:
        final_path += folder
        final_path += r"/"
    return final_path
