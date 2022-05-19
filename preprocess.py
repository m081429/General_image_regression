import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
global status


class Preprocess:

    def __init__(self, input_file_path, loss_function=None):
        """
        Return a randomized list of each directory's contents

        :param input_file_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """

        logger.debug('Initializing Preprocess')
        self.input_file_path = input_file_path
        self.loss_function = loss_function

        self.files, self.labels, self.min_images = self.__get_lists()

            
    def __get_lists(self):
        logging.debug('Getting initial list of images and labels')

        files = []
        labels = []
        min_images = 0
        fobj = open(self.input_file_path)
        for row in fobj:
            row=row.strip()
            columns = row.split(" ")
            columns[0]=columns[0].replace('sub_dir_h_e/','')
            labels.append(float(columns[1])*100)
            files.append(columns[0])
            min_images = min_images+1
        fobj.close()

        labels = tf.dtypes.cast(labels, tf.float32)
        return files, labels, min_images

#this is required to do additional data augmentation for training
def update_status(stat):
    global status
    status = stat
    return stat


# processing images
def format_example(image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :return: image
    """
    global status
    train = status
    image = tf.io.read_file(image_name)
    #image = tf.io.decode_jpeg(image)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)/255
    #image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))

    if train is True:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2,seed=44)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5,seed=44)
        image = tf.clip_by_value(image,0.0,1.0)
        #image = tf.image.random_jpeg_quality(image, min_jpeg_quality=20, max_jpeg_quality=90)
        #image = tf.keras.preprocessing.image.random_shear(image, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        #image = tf.keras.preprocessing.image.random_zoom(image, 0.9, row_axis=0, col_axis=1, channel_axis=2)
    image =tf.image.rgb_to_hsv(image)
    image = tf.reshape(image, (img_size, img_size, 3))

    return image


