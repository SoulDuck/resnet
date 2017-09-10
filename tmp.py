import tensorflow as tf
from sys import argv
flags=tf.app.flags
FLAGS=flags.FLAGS
import numpy as np
flags.DEFINE_string('test' , '' , '')

import six



a = [1, 2, 3]
a=np.asarray(a)
tensor_a = tf.Variable(a)

tensor_size=tf.size(tensor_a)
tensor_b =tf.expand_dims(tensor_a , 1)
tensor_range=tf.range(0,tensor_size,1)
tensor_range=tf.cast(tensor_range , tf.int64)
tensor_indices=tf.expand_dims(tensor_range,1)
tensor_concat=tf.concat([tensor_indices , tensor_b] ,1 )
onehot=tf.sparse_to_dense(tensor_concat ,[tensor_size,10] ,1.0 , 0.0  )
init=tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print sess.run(tensor_a )
print sess.run(tensor_size)
print sess.run(tensor_b)
print sess.run(tensor_range)
print sess.run(tensor_indices)
print sess.run(tensor_concat)
print sess.run(onehot)
def main(_):
    print 'parameters' , argv
    print 'input',FLAGS.test



label=tf.Variable(3)
onehot=tf.sparse_to_dense(label , [10] , 1.0 , 0.0 )
init=tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print sess.run(label)
print sess.run(onehot)
print



