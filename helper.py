import re
import random
import os.path
import shutil
from glob import glob

import numpy as np
import scipy.misc
import zipfile
import time
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm
from natsort import natsorted
from sklearn.utils import shuffle
import cv2


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


#def gen_batch_function(data_folder, image_shape):
#    """
#    Generate function to create batches of training data
#    :param data_folder: Path to folder that contains all the datasets
#    :param image_shape: Tuple - Shape of image
#    :return:
#    """
#    def get_batches_fn(batch_size):
#        """
#        Create batches of training data
#        :param batch_size: Batch Size
#        :return: Batches of training data
#        """
#        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
#        label_paths = {
#            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
#            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
#        background_color = np.array([255, 0, 0])
#
#        random.shuffle(image_paths)
#        for batch_i in range(0, len(image_paths), batch_size):
#            images = []
#            gt_images = []
#            for image_file in image_paths[batch_i:batch_i+batch_size]:
#                gt_image_file = label_paths[os.path.basename(image_file)]
#
#                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
#                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
#
#                gt_bg = np.all(gt_image == background_color, axis=2)
#                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#
#                images.append(image)
#                gt_images.append(gt_image)
#
#            yield np.array(images), np.array(gt_images)
#    return get_batches_fn


def gen_batch_function(image_paths, label_paths, image_shape, 
                       image_limit=0, 
                       augment=False,
                       exposure_shift=0.1,
                       shade_prob=0.25,
                       shade_intensity=(0.3,0.7)):
    """
    Generate function to create batches of training data.
    
    :param image_paths: image file paths
    :param label_paths: ground truth image file paths
    :param image_shape: Tuple - Shape of image
    :param image_limit: maximum number of images to yield (setting to 0 will yield all)
    :param augment: If true, apply random transformations to augment dataset.
    :param exposure_shift: shift image intensity mean by a random value between
        +/- exposure_shift
    :param shade_prob: probability of applying random shadow
    :param shade_intensity: (min, max) tuple of shadow intensity (between 0 and 1)

    :return get_batches_fn
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        background_color = np.array([255, 0, 0])

        N = len(image_paths)
        if image_limit > 0:
            N = min(N, image_limit)
        
        # shuffle images
        image_paths_shuffle, label_paths_shuffle = shuffle(image_paths, label_paths)
        
        for batch_i in range(0, N, batch_size):
            images = []
            gt_images = []
            for image_file, label_file in zip(image_paths_shuffle[batch_i:batch_i+batch_size], 
                                              label_paths_shuffle[batch_i:batch_i+batch_size]):
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(label_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                
                if augment:
                    # randomly flip images horizontally
                    if random.choice([True, False]):
                        image = np.fliplr(image)
                        gt_image = np.fliplr(gt_image)

                    # apply random shade
                    if np.random.uniform(0, 1) > (1 - shade_prob):
                        image = random_shade(image, intensity=shade_intensity)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(output_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
#    # Make folder for current run
#    output_dir = os.path.join(runs_dir, str(time.time()))
#    if os.path.exists(output_dir):
#        shutil.rmtree(output_dir)
#    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        
        
def train_val_split(data_dir, val_frac=0.1):
    """
    Split data into train and validation sets.
    """
    groups = ['um', 'umm', 'uu']
    imgs_train = []
    gt_imgs_train = []
    imgs_val = []
    gt_imgs_val = []
    for group in groups:
        imgs = natsorted(glob(os.path.join(data_dir, 'image_2', '{}_*'.format(group))))
        gt_imgs = natsorted(glob(os.path.join(data_dir, 'gt_image_2', '{}_road*'.format(group))))
        n_all = len(imgs)
        n_val = int(n_all * val_frac)
        n_train = n_all - n_val
        imgs_train.extend(imgs[:n_train])
        imgs_val.extend(imgs[n_train:])
        gt_imgs_train.extend(gt_imgs[:n_train])
        gt_imgs_val.extend(gt_imgs[n_train:])
    return imgs_train, imgs_val, gt_imgs_train, gt_imgs_val


# Data Augmentation Functions

def random_shade(img, intensity=(0.3,0.7)):
    """
    Apply random quadrilateral shadow to image.

    Shadow is bounded by 2 points along the top edge and 2 points along the 
    bottom edge of the image.
    
    :param img: input image
    :param intensity: (min, max) tuple defining random intensity range
    :returns shaded: image with random shadow
    """
    w = np.shape(img)[1]
    h = np.shape(img)[0]
    x_top = np.sort(np.random.uniform(0, w - 1, 2))
    x_bot = np.sort(np.random.uniform(0, w - 1, 2))
    corners = np.array([[[x_top[0], 0],
                       [x_top[1], 0],
                       [x_bot[1], h - 1],
                       [x_bot[0], h - 1]]], dtype=np.int32)
    mask = np.ones(np.shape(img), dtype=np.uint8)
    cv2.fillPoly(mask, corners, (0,)*3)
    mask = np.float32(mask)
    mask[mask==0] = np.random.uniform(intensity[0], intensity[1])
    shaded = img.copy()
    shaded = np.uint8(mask * shaded)
    return shaded