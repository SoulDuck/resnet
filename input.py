import tensorflow as tf

def build_input(dataset , data_path , batch_size , mode):
    image_size=32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0 #label offset =0 main cifar10
        n_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1 #label offset =1 mena cifar 100
        n_classes = 100
    else:
        raise ValueError('Not supported dataset')

    depth =3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes
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
    if mode == 'train':
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
            shapes = [[image_size , image_size , depth] , [1]])
        num_threads=16
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3*batch_size ,
            dtypes = [tf.float32 , tf.int32],
            shapes = [[image_size , image_size , depth] , [1]])
        num_threads = 1

    example_enqueue_op = example_queue.enqueue([image ,label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue , [example_enqueue_op]))
    # Read 'batch' labels + images from the example queue

    images , labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels , [batch_size , 1])
    indices = tf.reshape(tf.range(0 , batch_size , 1), [batch_size ,1 ])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices , labels] ,axis=1),
        [batch_size , n_classes] , 1.0 , 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) ==2 , len(labels.get_shape())
    assert labels.get_shape()[0] == batch_size , labels.get_shape()[0]
    assert labels.get_shape()[1] == n_classes  , labels.get_shape()[1]

    tf.summary.histogram('labels' , labels)
    tf.summary.image('images' , images )
    return images , labels





if '__main__' == __name__:
    image_size=32

    label_bytes = 1
    label_offset = 0 #label offset =0 main cifar10
    n_classes = 10

    depth =3
    batch_size=64
    image_bytes = image_size * image_size * depth

    record_bytes = label_bytes + label_offset + image_bytes
    print record_bytes
    print image_bytes
    print label_bytes
    print label_offset
    data_path='cifar-10/data_batch*'
    data_files = tf.gfile.Glob(data_path)

    file_queue = tf.train.string_input_producer(data_files , shuffle=True)
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    _ , value = reader.read(file_queue)
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record , [label_offset + label_bytes] , [image_bytes]) , [depth , image_size , image_size])

    #Convert from [ch , h , w ] to [h,w,ch]
    image = tf.cast(tf.transpose(depth_major , [1,2,0]) , dtype =  tf.float32)
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