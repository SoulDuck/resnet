import sys ,time , os
import numpy as np
import input
import tensorflow as tf
import six
import model
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset' , 'cifar10' , 'cifar-10 or cifar-100')
tf.app.flags.DEFINE_string('mode', 'train','train or eval')
tf.app.flags.DEFINE_string('train_data_path','cifar-10/data_batch*','Filepattern for training data')
tf.app.flags.DEFINE_string('eval_data_path' , 'cifar-10/test_batch.bin' , 'Filepatter for eval data')
tf.app.flags.DEFINE_integer('image_size', 32 , 'Image side length')
tf.app.flags.DEFINE_string('train_dir','./output/train','Directory to keep training outputs')
tf.app.flags.DEFINE_string('eval_dir','./output/eval', 'Directory to keep eval outputs')
tf.app.flags.DEFINE_integer('eval_batch_count',50,'Number of batches to eval')
tf.app.flags.DEFINE_string('f', './output/eval' ,'Directory to keep eval outputs' )
tf.app.flags.DEFINE_bool('eval_once' ,False , 'Whether evaluate the model only once')
tf.app.flags.DEFINE_string('log_root','./output','Directory to keep the checkpoints. Should be a parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus',1, 'Number of gpus used for training. (0 or 1 )')


def train(hps):
    """
    :param hparams:
    :return:
    """
    """training loop"""
    images , labels = input.build_input(FLAGS.dataset , FLAGS.train_data_path , hps.batch_size , FLAGS.mode)
    cls_resnet=model.resnet(hps , images , labels , FLAGS.mode) #initialize class resnet
    cls_resnet.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(), tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS) #this function for profiling
    sys.stdout.write('total_params: %d\n' *param_stats.total_parameters)

    truth = tf.argmax(cls_resnet.label , axis=1) # onehot --> cls
    predictions = tf.argmax(cls_resnet.predictions , axis=1) #onehot --> cls
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions , truth))) #mean average


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


def eval(hps):
    images, labels = input.build_input(FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
    cls_resnet=model.resnet(hps , images , labels , FLAGS.mode) #initialize class resnet
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



def main(_):
    if FLAGS.num_gpus==0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus==1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only suppoert 0 or 1 gpu')


    if FLAGS.mode == 'train':
        batch_size =128
    elif FLAGS.mode =='eval':
        batch_size = 100


    if FLAGS.dataset == 'cifar10':
        n_classes = 10
    elif FLAGS.dataset == 'cifar100':
        n_classes = 100


    hps=model.HParams(batch_size= batch_size ,
                     n_classes=n_classes ,
                     min_lrn_rate=0.0001 ,
                     lrn_rate=0.1 ,
                     n_residual_units=5 ,
                     use_bottleneck=False,
                     weight_decay_rate=0.0002,
                     relu_leakiness=0.1,
                     optimizer='mom')
    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode =='eval':
            eval(hps)

        #elif FLAGS.mode == 'eval':
        #    evaluate(hparams)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

