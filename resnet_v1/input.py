import tensorflow as tf


def _read_one_example( tfrecord_path , resize=(300,300)):
    tfrecord_paths=tf.gfile.Glob(tfrecord_path)
    print tfrecord_paths
    filename_queue = tf.train.string_input_producer(tfrecord_paths , num_epochs=100000)
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
    image=tf.reshape(image ,  image_shape)
    if not resize == None :
        resize_height , resize_width  = resize
        image_size_const = tf.constant((resize_height , resize_width , 3) , dtype = tf.int32)
        image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=resize_height,
                                               target_width=resize_width)
    #images  = tf.train.shuffle_batch([image ] , batch_size =batch_size  , capacity =30 ,num_threads=3 , min_after_dequeue=10)
    return image,label


def build_input(dataset , data_path , batch_size , mode):
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0 #label offset =0 main cifar10
        n_classes = 10
        depth =3
        image_size=32
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1 #label offset =1 mena cifar 100
        n_classes = 100
        depth =3
        image_size = 32
    elif dataset == 'fundus_300x300':
        n_classes =2
        depth =3
        image_size = 224
    else:
        raise ValueError('Not supported dataset')


    if dataset == 'cifar10' or dataset == 'cifar100':

        image_bytes = image_size * image_size * depth
        record_bytes = label_bytes + label_offset + image_bytes
        print data_path
        data_files = tf.gfile.Glob(data_path)
        print 'data files : ', data_files
        file_queue = tf.train.string_input_producer(data_files , shuffle=True)
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        _ , value = reader.read(file_queue)

        #convoert these examples to dense labels , and processed images
        record = tf.reshape(tf.decode_raw(value , tf.uint8),[record_bytes])
        label = tf.cast(tf.slice(record , [label_offset] , [label_bytes]) , tf.int32)

        #Convert from string to [depth * height * width ] to [depth , height , width]
        depth_major = tf.reshape(tf.slice(record , [label_offset + label_bytes] , [image_bytes]) , [depth , image_size , image_size])

        #Convert from [ch , h , w ] to [h,w,ch]
        image = tf.cast(tf.transpose(depth_major , [1,2,0]) , dtype =  tf.float32)
    elif dataset=='fundus_300x300':
        image , label  =_read_one_example('../sample_tfrecord/data_batch*', (image_size,image_size))



    if mode == 'train':
        image= tf.divide(image , 255 )
        image = tf.image.resize_image_with_crop_or_pad(image , image_size+4 , image_size+4)
        image = tf.random_crop(image , [image_size , image_size ,3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Brightness / saturatio / constrast provides samll gains 2%~5% on cifar

        image = tf.image.random_brightness(image , max_delta=63. / 255.)
        image = tf.image.random_saturation(image , lower=0.5 , upper=1.8)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8*batch_size,
            dtypes = [tf.float32 , tf.int32],
            shapes = [[image_size , image_size , depth] , [1] ])
        num_threads=16
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3*batch_size ,
            dtypes = [tf.float32 , tf.int32 , ],
            shapes = [[image_size , image_size , depth] , [1] ,])

        num_threads = 1
    label=tf.reshape(label , [1])
    print label
    example_enqueue_op = example_queue.enqueue([image ,label ])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue , [example_enqueue_op]))
    # Read 'batch' labels + images from the example queue


    images , labels  = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels , [batch_size , 1])
    indices = tf.reshape(tf.range(0 , batch_size , 1), [batch_size ,1 ])
    images, labels = example_queue.dequeue_many(batch_size)


    tf.summary.histogram('labels' , labels)
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices , labels] ,axis=1),
        [batch_size , n_classes] , 1.0 , 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) ==2 , len(labels.get_shape())
    assert labels.get_shape()[0] == batch_size , labels.get_shape()[0]
    assert labels.get_shape()[1] == n_classes  , labels.get_shape()[1]



    tf.summary.image('images' , images)
    return images , labels


if '__main__' == __name__:


    image_size=300

    label_bytes = 1
    label_offset = 1 #label offset =0 main cifar10
    n_classes = 10

    depth =3
    batch_size=128
    image_bytes = image_size * image_size * depth

    record_bytes = label_bytes + label_offset + image_bytes
    print record_bytes
    print image_bytes
    print label_bytes
    print label_offset
    data_path='cifar-10/data_batch*'
    data_files = tf.gfile.Glob(data_path)
    print data_files[0:2]
    file_queue = tf.train.string_input_producer(data_files[0:2] , shuffle=True)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)


    _ , value = reader.read(file_queue)
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record , [label_offset + label_bytes] , [image_bytes]) , [depth , image_size , image_size])
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    coord= tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess= sess ,coord = coord)

    try:
        count=0
        while not coord.should_stop():
            print len(sess.run(value))
            count += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        print count
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    sess.close()



    """
    #Convert from [ch , h , w ] to [h,w,ch]
    #image = tf.cast(tf.transpose(depth_major , [1,2,0]) , dtype =  tf.float32)
    example_queue = tf.FIFOQueue(
        3*batch_size ,
        dtypes = [tf.float32 , tf.int32],
        shapes = [[image_size , image_size , depth] , [1]])
    num_threads = 1
    example_enqueue_op = example_queue.enqueue([image ,label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue , [example_enqueue_op]))
    images, labels = example_queue.dequeue_many(batch_size)
    indices = tf.reshape(tf.range(0 , batch_size , 1), [batch_size ,1 ])

    indices = tf.sparse_to_dense(
        tf.concat(values=[indices , labels] ,axis=1),
        [batch_size , n_classes] , 1.0 , 0.0 )
    print record
    print label
    print value
    print image
    print indices
    """