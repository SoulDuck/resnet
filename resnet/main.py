import sys
import time
import random
import numpy as np
import six
import tensorflow as tf
import data
import input
import model
def divide_images_labels_from_batch(images, labels ,batch_size):
    debug_flag=False

    batch_img_list=[]
    batch_lab_list = []
    share=len(labels)/batch_size
    #print len(images)
    #print len(labels)
    #print 'share :',share

    for i in range(share+1):
        if i==share:
            imgs = images[-batch_size:]
            labs = labels[-batch_size:]
            #print i+1, len(imgs), len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
            if __debug__ ==debug_flag:
                print "######utils.py: divide_images_labels_from_batch debug mode#####"
                print 'total :', len(images), 'batch', i*batch_size ,'-',len(images)
        else:
            imgs=images[i*batch_size:(i+1)*batch_size]
            labs=labels[i * batch_size:(i + 1) * batch_size]
           # print i , len(imgs) , len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
            if __debug__ == debug_flag:
                print "######utils.py: divide_images_labels_from_batch debug mode######"
                print 'total :', len(images) ,'batch' ,i*batch_size ,":",(i+1)*batch_size
    return batch_img_list , batch_lab_list
FLAGS=tf.app.flags.FLAGS
dataset='cifar10'

dataset='fundus_300x300'
dataset='mnist'
if dataset == 'cifar10':
    image_size=32
    depth=3
elif dataset=='fundus_300x300':
    image_size=300
    depth=3
elif dataset=='mnist':
    image_size=28
    depth=1
else:
    raise AssertionError

tf.app.flags.DEFINE_string('dataset' , dataset , 'cifar-10 or cifar-100 or mnist or fundus_300x300' )
tf.app.flags.DEFINE_string('mode', 'train','train or eval')
tf.app.flags.DEFINE_string('train_data_path','../'+dataset+'/data_batch*','Filepattern for training data')
tf.app.flags.DEFINE_string('eval_data_path' , '../'+dataset+'/test_batch.bin' , 'Filepatter for eval data')
tf.app.flags.DEFINE_integer('image_size', image_size , 'Image side length')
tf.app.flags.DEFINE_string('train_dir','./output/train','Directory to keep training outputs')
tf.app.flags.DEFINE_string('eval_dir','./output/eval', 'Directory to keep eval outputs')
tf.app.flags.DEFINE_integer('eval_batch_count',50,'Number of batches to eval')
tf.app.flags.DEFINE_string('f', './output/eval' ,'Directory to keep eval outputs' )
tf.app.flags.DEFINE_bool('eval_once' ,False , 'Whether evaluate the model only once')
tf.app.flags.DEFINE_string('log_root','./output','Directory to keep the checkpoints. Should be a parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus',1, 'Number of gpus used for training. (0 or 1 )')


def next_batch(imgs, labs, batch_size):
    indices = random.sample(range(np.shape(imgs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    return batch_xs, batch_ys

def train(hps):
    """
    :param hparams:
    :return:
    """
    """training loop"""
    """
     self.hps = batch_size= batch_size,
                n_classes=n_classes,
                min_lrn_rate=0.0001,
                lrn_rate=0.1,
                n_residual_units=5,
                use_bottleneck=False,
                weight_decay_rate=0.0002,
                relu_leakiness=0.1,
                optimizer='mom')
     self._images = images
     self.label = labels
     self.mode = mode
     self._extra_train_ops = []
     self.predictions  = tf.nn.softmax(logits)
     self.lrn_rate
     self.train_op
     self.summaries
     """



    #images , labels=data.fundus_np_load()
    images  , labels, test_imgs , test_labs, = data.mnist()
    x_ = tf.placeholder(dtype=tf.float32 , shape=[hps.batch_size , image_size, image_size ,depth])
    y_ = tf.placeholder(dtype=tf.int32, shape=[hps.batch_size])
    lrn_ = tf.placeholder(dtype=tf.int32)
    onehot = tf.one_hot(y_ , depth=hps.n_classes)

    cls_resnet= model.resnet(hps, x_, onehot, FLAGS.mode) #initialize class resnet
    cls_resnet.build_graph()
    print 'build graph done'
    #cls_resnet class Variable

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS) #this function for profiling
    print param_stats.total_parameters
    sys.stdout.write('total_params: %d\n' *param_stats.total_parameters)


    truth = tf.argmax(cls_resnet.label , axis=1) # onehot --> cls
    predictions = tf.argmax(cls_resnet.predictions , axis=1) #onehot --> cls
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions , truth))) #mean average
    #summary_op = tf.summary.merge([cls_resnet.summaries, tf.summary.scalar('Precision', precision)])
    #tfboard_writer = tf.summary.FileWriter(FLAGS.train_dir)

    init= tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    print 'a'
    sess=tf.Session()
    print 'b'
    sess.run(init)
    print 'c'
    batch_xs , batch_ys=next_batch(images , labels , hps.batch_size)
    print 'd'
    check_point=100

    for i in range(10000):
        msg = '\r Progress {0}/{1}'.format(i,10000)
        sys.stdout.write(msg)
        sys.stdout.flush()
        if i<1000:
            lrn_point=0.01
        elif i<2000:
            lrn_point = 0.001
        elif i < 5000:
            lrn_point = 0.0001
        else:
            lrn_point = 0.0001
        if i%check_point==0:
            imgs_ , labs_ =divide_images_labels_from_batch(test_imgs , test_labs , batch_size)
            imgs_labs=zip(imgs_ , labs_)
            sum_precision=0
            for i,(xs , ys) in enumerate(imgs_labs):
                sum_precision+=sess.run(precision, feed_dict={x_: xs, y_: ys , lrn_ :lrn_point })
            print sum_precision/float(i+1)

        sess.run(cls_resnet.train_op, feed_dict={x_: batch_xs, y_: batch_ys , lrn_ : lrn_point})


    print 'f'




    """

    summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir=FLAGS.train_dir,\
                                             summary_op=tf.summary.merge([cls_resnet.summaries, tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(tensors={'step': cls_resnet.global_step ,
                                                       'loss': cls_resnet.cost,
                                                       'precision':precision} , every_n_iter=100)


    class _LearningRateSetterHook(tf.train.SessionRunHook):
        def begin(self):
            self._lrn_rate = 0.1
        def before_run(self, run_context):
            return tf.train.SessionRunArgs(cls_resnet.global_step , feed_dict={cls_resnet.lrn_rate : self._lrn_rate})
        def after_run(self , run_context , run_values):
            train_step = run_values.results
            if train_step < 40000:
                self._lrn_rate=0.1
            elif train_step < 60000:
                self._lrn_rate = 0.01
            elif train_step < 80000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.0001

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = FLAGS.log_root ,
        hooks=[logging_hook , _LearningRateSetterHook()],
        chief_only_hooks=[summary_hook],
        save_summaries_steps=0,
        config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
     while not mon_sess.should_stop():
         mon_sess.run(cls_resnet.train_op)

    """
def eval(hps):
    images, labels = input.build_input(FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
    cls_resnet= model.resnet(hps, images, labels, FLAGS.mode) #initialize class resnet
    cls_resnet.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision=0.0

    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e :
            tf.logging.error('Cannot restore checkpoint : %s' ,e)
            continue
        if not ( ckpt_state and ckpt_state.model_checkpoint_path ):
            tf.logging.error('No model to eval yet at %s' ,FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s' ,ckpt_state.model_checkpoint_path)
        saver.restore(sess , ckpt_state.model_checkpoint_path)

        total_prediction , correct_prediction = 0,0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [cls_resnet.summaries, cls_resnet.cost, cls_resnet.predictions,
                 cls_resnet.label, cls_resnet.global_step])
            truth = np.argmax(truth , axis =1 )
            predictions =np.argmax(predictions , axis= 1 )
            correct_prediction +=np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0*correct_prediction/total_prediction
        best_precision = max(precision , best_precision)


        precision_summ = tf.Summary()
        precision_summ.value.add(tag='Precision' , simple_value=precision)
        summary_writer.add_summary(precision_summ , train_step)

        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(tag='Best Precision' , simple_value=best_precision)

        summary_writer.add_summary(best_precision_summ , train_step)
        summary_writer.add_summary(summaries ,train_step)
        tf.logging.info('loss:%.3f , precisions: %.3f , best_precision : %.3f' %(loss ,precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break;
        time.sleep(60)


if __name__ == '__main__':
    if FLAGS.num_gpus==0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus==1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only suppoert 0 or 1 gpu')


    if FLAGS.mode == 'train':
        batch_size =60
    elif FLAGS.mode =='eval':
        batch_size = 100


    if FLAGS.dataset == 'cifar10':
        n_classes = 10
    elif FLAGS.dataset == 'cifar100':
        n_classes = 100
    elif FLAGS.dataset == 'fundus_300x300':
        n_classes = 2
    elif FLAGS.dataset == 'mnist':
        n_classes = 10



    hps= model.HParams(batch_size= batch_size,
                       n_classes=n_classes,
                       min_lrn_rate=0.0001,
                       lrn_rate=0.1,
                       n_residual_units=5,
                       use_bottleneck=False,
                       weight_decay_rate=0.0002,
                       relu_leakiness=0.1,
                       optimizer='mom')
    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode =='eval':
            eval(hps)
    #    tf.logging.set_verbosity(tf.logging.INFO)
    #    tf.app.run()"""