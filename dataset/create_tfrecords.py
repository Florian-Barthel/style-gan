import glob
import tensorflow as tf
import PIL.Image
import numpy as np
import os
from tqdm import tqdm


def create_from_images(tfrecord_file, image_dir):
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for image_file in tqdm(image_filenames):
            img = np.asarray(PIL.Image.open(image_file))
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            writer.write(ex.SerializeToString())


create_from_images(tfrecord_file='E:/my_tfRecords/ffhq256.tfrecords', image_dir='E:/ffhq_256')
