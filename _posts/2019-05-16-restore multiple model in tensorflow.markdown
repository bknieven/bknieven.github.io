---
layout: post
title: "Tensorflow中加载多个model"
date: 2019-05-16 17:25:00 +0800
---

&emsp;&emsp;如果在restore参数时，初始化saver时没有指定载入的变量名，则会在checkpoint中为当前文件已定义的所有变量寻找保存的值，如果没有找到，则会报错。
所以，如果要restore两个model（也可认为是从两个checkpoint中restore当前session中的变量），即分两批restore，如果在第一个saver在restore第一个model的checkpoint文件时，已经定义了两个model，且这个saver没有指定仅仅restore第一个model的变量，那么就会报错，因为当前restore操作会在第一个model的checkpoint中寻找已定义的所有变量，包括第二个model的变量，但第一个checkpoint中只包含第一个model的变量。

如果要在一个session中restore两个model，或者可更普通的说restore两组变量，正确的做法应该是：
1. 在训练和保存两个model参数时，使用
`with tf.variable_scope('')`为两个model的变量定义不同的variable_scope，variable_scope会使存储的变量名为“variable_scope名/原始变量名”的形式；
2. 载入模型参数时，同样为两个model使用`with tf.variable_scope('')`定义与训练时相同的variable_scope；
3. 载入模型参数时，为两个model分别定义saver，并分别为两个saver定义所要处理的变量。具体来说，因为在模型训练时已经使用了`with tf.variable_scope('')`为两个model的变量定义了不同的variable_scope，这里就可以根据变量名中包含的variable_scope的字符串对变量进行筛选，这就是使用`with tf.variable_scope('')`的目的。虽然不使用variable_scope也可以在定义saver时手动列出两个model的变量，但使用variable_scope更加便捷，也可以避免两个model存在相同变量名的情况。另外，载入模型参数操作和使用几个session没有关系，即session不会影响变量的作用域。

参考：[https://github.com/tensorflow/tensorflow/issues/3270](https://github.com/tensorflow/tensorflow/issues/3270)

测试：

save_model1.py
~~~ python
import tensorflow as tf
with tf.variable_scope('model1'):
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
    
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './save/model1')
~~~

save_model2.py
~~~ python
import tensorflow as tf
with tf.variable_scope('model2'):
    w3 = tf.Variable(tf.random_normal(shape=[3]), name='w3')
    w4 = tf.Variable(tf.random_normal(shape=[4]), name='w4')
    
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './save/model2')
~~~

restore.py
~~~ python
import tensorflow as tf
model1_file = "./save/model1"
model2_file = "./save/model2"

sess = tf.Session()

with tf.variable_scope('model1'):
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
with tf.variable_scope('model2'):
    w3 = tf.Variable(tf.random_normal(shape=[3]), name='w3')
    w4 = tf.Variable(tf.random_normal(shape=[4]), name='w4')

saver1 = tf.train.Saver([v for v in tf.all_variables() if 'model1' in v.name])
saver1.restore(sess, model1_file)
print("w1 : %s" % w1.eval(sess))
print("w2 : %s" % w2.eval(sess))

saver2 = tf.train.Saver([v for v in tf.all_variables() if 'model2' in v.name])
saver2.restore(sess, model2_file)
print("w3 : %s" % w3.eval(sess))
print("w4 : %s" % w4.eval(sess))

print("Model restored.")
~~~

