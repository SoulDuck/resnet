#
"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
import six



HParams = namedtuple('HParams',
                     'batch_size, n_classes, min_lrn_rate, lrn_rate, '
                     'n_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')

#this param set in main


class resnet(object):
    def __init__(self, hparam, images ,  labels ,mode ):
        print 'resnet class init'
        """
        :param hparam: hyperparameter : learning rate , checkpoint ,
        :param images: tensor shape rank 4  | original shape
        :param labels: onehot | cls
        :param mode: train | eval
        :return:
        """
        self.hps = hparam
        self._images = images
        self.label = labels
        self.mode = mode
        self._extra_train_ops = []

    def _stride(self ,stride):
        """
        Note: if you want more detail strides option . input stride
        :param stride:
        :return:
        """
        return [1 , stride , stride , 1]


    def _conv(self , name , x , k_size , in_ch , out_ch, stride , _padding='SAME'):
        """

        :param name: tensor name
        :param x_: input tensor(placeholder)
        :param filter_size:
        :param input_filters:
        :param output_filters:
        :param strides:
        :return:
        """
        n=k_size * k_size * in_ch
        k= tf.get_variable('conv_w' ,[k_size , k_size , in_ch , out_ch] , dtype = tf.float32 , \
                           initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x , k , stride , padding=_padding)

    def _act(self , x , actmode='relu'):
        if actmode =='relu':
            return tf.nn.relu(x , name='relu')
        elif actmode == 'leaky_relu':
            leakiness=0.05 # how about set leakiness randomly?
            return tf.where(tf.less(x,0.0) , self.hps.relu_leakiness * x , x  , name ='leaky_relu')

        return tf.where(tf.less)
        """
        relu | elu | sigmoid |
        :return:
        """

    def _batch_norm(self , name , x):
        """

        :param name:
        :param x:
        :return:
        """
        p_shape=[x.get_shape()[-1]]

        with tf.variable_scope(name):
            beta = tf.get_variable(name='beta', shape=p_shape, dtype=tf.float32, \
                            initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)


            gamma = tf.get_variable(name='gamma', shape=p_shape, dtype=tf.float32, \
                        initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)


        if self.mode == 'train':

                    mean , variance =tf.nn.moments(x , [0,1,2] , name = 'momnets')
                    moving_mean = tf.get_variable('moving_mean',shape=p_shape, dtype=tf.float32 ,\
                                                  initializer=tf.constant_initializer(0.0 , tf.float32))
                    moving_variance = tf.get_variable(name='moving_variance', shape=p_shape , dtype=tf.float32 ,\
                                                      initializer=tf.constant_initializer(1.0 , tf.float32))

                    self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean , mean , 0.9))
                    self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance , variance , 0.9))

        else:
            mean = tf.get_variable(name = 'moving_mean' ,shape = p_shape , dtype = tf.float32 ,\
                                  initializer=tf.constant_initializer(0.0 , tf.float32), trainable=False)
            variance = tf.get_variable(name = 'moving_variance', shape = p_shape , dtype = tf.float32 , \
                                       initializer=tf.constant_initializer(1.0 , tf.float32) , trainable=False)

            tf.summary.histogram(mean.op.name , mean)
            tf.summary.histogram(variance.op.name , variance)

        y=tf.nn.batch_normalization(x, mean , variance , beta , gamma , 0.001)
        y.set_shape(x.get_shape())
        return y


    def _affine(self ,  x ,  out_ch):
        x=tf.reshape(x,[self.hps.batch_size , -1 ])
        w=tf.get_variable('fc_w' ,[x.get_shape()[1] , out_ch] , initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b=tf.get_variable('fc_b' ,[out_ch] , initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x,w,b)


    def _gap(self , x ):
        assert x.get_shape().ndims==4
        return tf.reduce_mean(x , [1,2])


    def _residual(self , x , in_ch , out_ch , stride , act_before_residual=False ):
        if act_before_residual==True:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn' , x )
                x = self._act(x , 'leaky_relu')
                orig_x = x
                #print x.get_shape()
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn',x)
                x = self._act(x , 'leaky_relu')
                #print x.get_shape()
        with tf.variable_scope('sub1'):
            x= self._conv('conv1' , x , 3 ,in_ch , out_ch , stride)
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2' , x)
            x = self._act(x , 'leaky_relu')
            x= self._conv('conv2' , x , 3 , out_ch , out_ch ,[1,1,1,1])
        with tf.variable_scope('sub_add'):
            if in_ch != out_ch:
                print 'orig_x shape :',orig_x.get_shape()
                print 'x shape: ',x.get_shape()
                print 'in_ch : ',in_ch , 'out_ch : ' , out_ch
                orig_x = tf.nn.avg_pool(orig_x , stride , stride , 'VALID')
                print orig_x.get_shape()
                orig_x = tf.pad( orig_x, [[0,0],[0,0],[0,0], [(out_ch - in_ch)//2 , (out_ch-in_ch)//2]])
                print orig_x.get_shape()
                print 'a'

            x+=orig_x

            tf.logging.debug('image after unit %s' ,x.get_shape())
            return x
    def _decay(self):
        costs=[]
        for var in tf.trainable_variables():
            if var.op.name.find(r'conv_w') > 0:
                print var.op.name
                costs.append(tf.nn.l2_loss(var))
            if var.op.name.find(r'fc_w') > 0:
                print var.op.name
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate , tf.add_n(costs))

    def _build_model(self):
        with tf.variable_scope('init'):
            x_ = self._images
            x_ = self._conv('conv' , x_ , 3 ,3 ,16 , self. _stride(1))
            strides = [1, 1, 2]
            activate_before_residual = [True, False, False]

            if self.hps.use_bottleneck:
                res_func = self._bottlenect_residual
                filters= [16, 64,128 ,256]

            else:
                res_func = self._residual
                filters = [16,16,32,64]

        with tf.variable_scope('unit_1'):
            x_= res_func(x_,filters[0] , filters[1] ,self._stride(strides[0]) , activate_before_residual[0])
        for i in six.moves.range(1  , self.hps.n_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x_=res_func(x_ , filters[1] , filters[1] , self. _stride(1) , False)
        print x_.get_shape()

        with tf.variable_scope('unit_2'):
            x_ = res_func(x_ , filters[1] , filters[2] , self._stride(strides[1]) ,activate_before_residual[1])
        print x_.get_shape()
        for i in six.moves.range(1, self.hps.n_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x_=res_func(x_, filters[2] , filters[2] ,self._stride(1) , False)
        print x_.get_shape()
        with tf.variable_scope('unit_3'):
            x_ = res_func(x_ , filters[2] , filters[3] , self._stride(strides[2]) , activate_before_residual[2])
        for i in six.moves.range(1, self.hps.n_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x_=res_func(x_ , filters[3] , filters[3] , self._stride(1) , False)
        print x_.get_shape()
        with tf.variable_scope('unit_last'):
            x_ = self._batch_norm('final_bn' , x_ )
            x_ = self._act(x_ ,actmode='leaky_relu')
            x_ = self._gap(x_)
        print x_.get_shape()
        with tf.variable_scope('logits'):
            logits = self._affine(x_ , self.hps.n_classes)
            self.predictions  = tf.nn.softmax(logits)
        with tf.variable_scope('cost'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits , labels=self.label)
            self.cost = tf.reduce_mean(xent,name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost',self.cost)


    def _build_train_op(self):
        """

        :return:
        """

        self.lrn_rate =tf.constant(self.hps.lrn_rate , tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate )

        trainable_variable= tf.trainable_variables()
        grads = tf.gradients(self.cost ,trainable_variable)

        if self.hps.optimizer== 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer=='mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate , 0.9)

            self.train_ops=optimizer.minimize(self.cost)
            #apply_op = optimizer.apply_gradients(zip(grads , trainable_variable) , global_step=self.global_step , name='train_step')
        #train_ops = [apply_op] + self._extra_train_ops
        #self.train_op = tf.group(*train_ops)

    def build_graph(self):
        """using thie function you can build more detail model"""
        """model arcitecture"""
        with tf.variable_scope('init'):
            """ in this line , start model"""
        x_ = self._images
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()





