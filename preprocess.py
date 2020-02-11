import tensorflow as tf
import numpy as np
import glob
import random
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image



def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'encoded': _bytes_feature(img),
               'label': _int64_feature(lab)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def write_tfrecord(imagedir, traindir, validdir, ratio):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    writer_train = tf.python_io.TFRecordWriter(traindir)
    if(ratio < 1):
        writer_valid = tf.python_io.TFRecordWriter(validdir)

    filenames = glob.glob(imagedir+'/*/*')
    f_all = []
    for f in filenames:
        f_all.append(f)
    random.shuffle(f_all)
    f_train = f_all[:int(len(f_all) * ratio)]
    f_valid = f_all[int(len(f_all) * ratio):]
    for f in f_train:
        lab = int(f[-11])
        img_data = _image_as_bytes(f)
        example = make_example(img_data, lab)
        writer_train.write(example)
    for f in f_valid:
        lab = int(f[-11])
        img_data = _image_as_bytes(f)
        example = make_example(img_data, lab)
        writer_valid.write(example)

    '''
    for l in labels:
        lab = int(l[-1:])
        print(lab)
        filenames = glob.glob(l+'/*')
        f_all = [];
        for f in filenames:
            f_all.append(f);

        f_train = f_all[:int(len(f_all)*ratio)]
        f_valid = f_all[int(len(f_all)*ratio):]

        for f in f_train:
            img_data = _image_as_bytes(f)
            example = make_example(img_data,lab)
            writer_train.write(example)
        for f in f_valid:
            img_data = _image_as_bytes(f)
            example = make_example(img_data,lab)
            writer_valid.write(example)

        print(str(len(f_train))+" "+str(len(f_valid))+" "+str(len(f_all)))
    '''

    writer_train.close()
    if(ratio<1):
        writer_valid.close()




def read_tfrecord(folder, batch=100, epoch=1):

    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """
    filename = folder+'\mnist.tfrecord'
    filename_queue = tf.train.string_input_producer([filename], epoch)


    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                      'label': tf.FixedLenFeature([],tf.int64, default_value=0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    img = tf.decode_raw(features['encoded'], tf.uint8)



    #28,28,1
    img = tf.reshape(img,[28, 28, 1])
    img = tf.cast(img, tf.float32) * (1. / 255)
    lab = features['label']
    lab = tf.one_hot(lab, 10)


    min_after_dequeue = 10
    img_batch, lab_batch = tf.train.shuffle_batch([img, lab], batch_size=batch, capacity=min_after_dequeue+3*batch,
                                                min_after_dequeue=min_after_dequeue)


    return img_batch, lab_batch