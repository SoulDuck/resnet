import tensorflow as tf
import numpy as np


def _read_one_example( tfrecord_path , resize=(300,300)):
    tfrecord_paths=tf.gfile.Glob(tfrecord_path+'/*.tfrecord')
    print 'tfrecord paths :',tfrecord_paths
    filename_queue = tf.train.string_input_producer(tfrecord_paths , num_epochs=100)
    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64),
        'filename': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    filename = tf.cast(features['filename'] , tf.string)

    image_shape = tf.stack([height , width , 3 ])
    print 'image shape : ',image_shape
    image=tf.reshape(image ,  image_shape)
    print 'image ',image
    if not resize == None :
        resize_height , resize_width  = resize
        image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=resize_height,
                                               target_width=resize_width)
    #images  = tf.train.shuffle_batch([image ] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
    return image,label , filename


"""
def get_shuffled_batch(tfrecord_path , batch_size ):
    image , label =_read_one_example(tfrecord_path)

    image_size=int(image.get_shape()[1])
    depth = int(image.get_shape()[2])
    print 'image size : ',image_size
    print 'depth : ', depth
    example_queue=tf.FIFOQueue(3*batch_size ,
            dtypes = [tf.uint8 , tf.int32  ],
            shapes = [[image_size , image_size ] , [] ])
    num_threads = 1
    print image
    print label
    print filename
    example_queue_op=example_queue.enqueue([image , label , filename])
    print '#'

    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue , [example_queue_op]))
    images, labels , filenames = example_queue.dequeue_many(batch_size)
    return images , labels , filenames

"""

def get_shuffled_batch( tfrecord_path , batch_size , resize ):
    resize_height , resize_width  = resize
    print tfrecord_path
    filename_queue = tf.train.string_input_producer(tfrecord_path , num_epochs=10)

    reader = tf.TFRecordReader()
    _ , serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'raw_image': tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([] , tf.int64),
        'filename': tf.FixedLenFeature([] , tf.string)
        })
    image = tf.decode_raw(features['raw_image'], tf.uint8)
    height= tf.cast(features['height'] , tf.int32)
    width = tf.cast(features['width'] , tf.int32)
    label = tf.cast(features['label'] , tf.int32)
    filename = tf.cast(features['filename'] , tf.string)

    image_shape = tf.stack([height , width , 3 ])  #image_shape shape is ..
    image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
    image=tf.reshape(image ,  image_shape)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=resize_height,
                                           target_width=resize_width)
    images  , labels  , filenames= tf.train.shuffle_batch([image ,label , filename] , batch_size =batch_size  , capacity =30 ,num_threads=1 , min_after_dequeue=10)
    return images  ,labels , filenames


if '__main__' == __name__:
    tfrecord_paths = tf.gfile.Glob('sample_tfrecord'+'/*.tfrecord')
    print tfrecord_paths
    images , labels , filenames=get_shuffled_batch(tfrecord_paths , batch_size=60 , resize=(299,299))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in xrange(2):
        imgs, labs , fnames_= sess.run([images, labels , filenames])
        print np.shape(imgs)
        print labs
        print fnames_
    coord.request_stop()
    coord.join(threads)

    """
    try:
        count = 0
        while not coord.should_stop():
            image , label , filename =sess.run(fetches=[image , label , filename])
            count += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        print count
        # When done, ask the threads to stop.
        coord.request_stop()
    """



    """
    images , labels,filenames =get_shuffled_batch(tfrecord_path , batch_size=60 , resize=(300,300))



    init=tf.group(tf.local_variables_initializer() ,tf.global_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    coord= tf.train.Coordinator()
    threads =tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        _images , _labels  , _filenames= sess.run([images , labels , filenames])

    coord.request_stop()
    coord.join(threads)
    """
