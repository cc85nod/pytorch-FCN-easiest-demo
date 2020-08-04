import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    shear_range=0.1,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

DATAS_PATH='./data'
TRAINING_PATH=f'{DATAS_PATH}/training'
TESTING_PATH=f'{DATAS_PATH}/testing'

TRAINING_IMG_PATH=f'{TRAINING_PATH}/images'
TRAINING_LABEL_PATH=f'{TRAINING_PATH}/labels'
TESTING_IMG_PATH=f'{TESTING_PATH}/images'

EXAMPLE_PATH='./examples/'
EX_CLASSES = [
    'A',
    'B',
]

def create_dirs():
    # base dir
    if not os.path.exists(DATAS_PATH):
        os.makedirs(DATAS_PATH)
    if not os.path.exists(TRAINING_PATH):
        os.makedirs(TRAINING_PATH)
    if not os.path.exists(TESTING_PATH):
        os.makedirs(TESTING_PATH)

    # img and label dir
    if not os.path.exists(TRAINING_IMG_PATH): # training img
        os.makedirs(TRAINING_IMG_PATH)
    if not os.path.exists(TRAINING_LABEL_PATH): # training label
        os.makedirs(TRAINING_LABEL_PATH)
    if not os.path.exists(TESTING_IMG_PATH): # testing img
        os.makedirs(TESTING_IMG_PATH)

""" Image Generator

width x height: image size
shape: circle, rectangle, ...
nums: how many image will be generate
datatype: train, test or validation
"""
def generate_img(width, height, nums):
    create_dirs()

    for cname in EX_CLASSES:
        img_generator = datagen.flow_from_directory(
            directory=EXAMPLE_PATH+cname+'/images',
            target_size=(width, height),
            batch_size=1,
            classes=None,
            class_mode=None,
            subset='training',
            save_to_dir=TRAINING_IMG_PATH,
            save_prefix=cname,
            shuffle=False,
            seed=1,
        )
        label_generator = datagen.flow_from_directory(
            directory=EXAMPLE_PATH+cname+'/labels',
            target_size=(width, height),
            batch_size=1,
            classes=None,
            class_mode=None,
            subset='training',
            save_to_dir=TRAINING_LABEL_PATH,
            save_prefix=cname,
            shuffle=False,
            seed=1,
        )
        
        for _ in range(nums // len(EX_CLASSES)):
            img_generator.next()
            label_generator.next()

generate_img(*IMAGE_SIZE, GEN_IMAGE_NUM)