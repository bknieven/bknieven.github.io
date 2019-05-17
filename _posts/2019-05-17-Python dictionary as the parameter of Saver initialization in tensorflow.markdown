---
layout: post
title: "Tensorflow中使用Python字典作为初始化Saver的参数"
date: 2019-05-17 11:34:00 +0800
---

&emsp;&emsp;Saver初始化时可以接收一个变量列表或者Python字典，来控制处理哪些变量，如果不指定，则默认处理目前已定义的所有变量（即，当前Saver初始化语句作用域范围内的所有变量）。

> * 变量列表（将以其本身的名称保存）。
> * Python 字典，其中，键是要使用的名称，键值是要管理的变量。

* 变量列表为**当前文件**中所定义的变量，而非checkpoint文件中存的变量名。
* Python字典中key为checkpoint文件中的变量名，value为**当前文件**中所定义的变量。

参考：[https://www.tensorflow.org/guide/saved_model](
https://www.tensorflow.org/guide/saved_model)

测试：

save_model.py
~~~ python
import tensorflow as tf

with tf.variable_scope('model'):
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
 
keys = ['model_x/w1','model_x/w2']
values = [v for v in tf.all_variables() if 'model' in v.name]
saver = tf.train.Saver(dict(zip(keys,values)))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './save/model')
~~~

restore_model.py
~~~ python
import tensorflow as tf

model1_file = "./save/model"
with tf.variable_scope('model_1'):
    w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
    w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

keys = ['model_x/w1','model_x/w2']
values = [v for v in tf.all_variables() if 'model_1' in v.name]
saver1 = tf.train.Saver(dict(zip(keys,values)))
sess = tf.Session()
saver1.restore(sess, model1_file)
print("w1 : %s" % w1.eval(sess))
print("w2 : %s" % w2.eval(sess))
~~~

&emsp;&emsp;在上面的实验中，模型存储部分和模型加载部分都用到的使用python字典作为参数的Saver，在实际中这种情况很少出现，一般只是在方使用这种映射实现模型的正常加载。

&emsp;&emsp;在上面实验中存储的checkpoint文件中，变量以变量名"model_x/..."的形式存储，而在save_model.py和restore_model.py中，由于使用的variable_scope的名字不同，导致定义的tensorflow变量名与"model_x/..."不同，但是使用Saver的字典参数，可以实现这个checkpoint文件能够被restore_model.py加载使用。