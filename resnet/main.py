import sys
import time
import random
import numpy as np
import six
import tensorflow as tf
import data
import input
import model


FLAGS=tf.app.flags.FLAGS
dataset='cifar10'
dataset='fundus_300x300'

if dataset == 'cifar10':
    image_size=32
elif dataset=='fundus_300x300':
    image_size=300
else:
    raise AssertionError

tf.app.flags.DEFINE_string('dataset' , dataset , 'cifar-10 or cifar-100')
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
tf.app.flags.DEFINE_integer('num_gpus',0, 'Number of gpus used for training. (0 or 1 )')


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
    images , labels=data.fundus_np_load()
    x_ = tf.placeholder(dtype=tf.float32 , shape=[hps.batch_size , 300, 300 ,3])
    y_ = tf.placeholder(dtype=tf.int32, shape=[hps.batch_size])
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

    summary_op = tf.summary.merge([cls_resnet.summaries, tf.summary.scalar('Precision', precision)])
    tfboard_writer = tf.summary.FileWriter(FLAGS.train_dir)

    init= tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    batch_xs , batch_ys=next_batch(images , labels , hps.batch_size)
    sess.run([cls_resnet.train_op] , feed_dict = {x_ : batch_xs , y_: batch_ys })






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


"""

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