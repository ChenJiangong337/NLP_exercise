#   Filename:   TensorFlow.py

import tensorflow as tf

#   张量（Tensor）

t0 = tf.constant(3,dtype=tf.int32)
t1 = tf.constant([3.,4.1,5.2],dtype=tf.float32)
t2 = tf.constant([['Apple','Orange'],['Potato','Tomato']],dtype=tf.string)
t3 = tf.constant([[[5],[6],[7]],[[4],[3],[2]]])

print(t0)
print(t1)
print(t2)
print(t3)

sess = tf.Session()
print(sess.run(t0))
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t3))

#   构建计算图

node1 = tf.constant(3.2)
node2 = tf.constant(4.8)
adder = node1+node2
print(adder)
sess = tf.Session()
print(sess.run(adder))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(a)
print(b)
print(adder_node)
sess = tf.Session()
print(sess.run(adder_node,{a:3,b:4.5}))
print(sess.run(adder_node,{a:[1,3],b:[2,4]}))

add_and_triple = adder_node*3.
print(sess.run(add_and_triple,{a:3,b:4.5}))

#   TensorFlow 应用实例

W = tf.Variable([.1],dtype=tf.float32)
b = tf.Variable([-.1],dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(W))

print(sess.run(linear_model,{x:[1,2,3,6,8]}))

print(sess.run(loss,{x:[1,2,3,6,8],y:[4.8,8.5,10.4,21.0,25.3]}))

fixW = tf.assign(W,[2.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])
print(sess.run(loss,{x:[1,2,3,6,8],y:[4.8,8.5,10.4,21.0,25.3]}))

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

x_train = [1,2,3,6,8]
y_train = [4.8,8.5,10.4,21.0,25.3]

for i in range(10000):
    sess.run(train,{x:x_train,y:y_train})

print('W: %s b: %s loss: %s'%(sess.run(W),sess.run(b),sess.run(loss,{x:x_train,y:y_train})))

#   tensorboard

W = tf.Variable([.1],dtype=tf.float32)
b = tf.Variable([-.1],dtype=tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W*x+b

with tf.name_scope('loss-model'):
    loss = tf.reduce_sum(tf.square(linear_model - y))
    tf.summary.scalar('loss',loss)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

x_train = [1,2,3,6,8]
y_train = [4.8,8.5,10.4,21.0,25.3]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter('/tmp/tensorflow',sess.graph)

for i in range(10000):
    summary,_ = sess.run([merged,train],{x:x_train,y:y_train})
    writer.add_summary(summary,i)

curr_W,curr_b,curr_loss = sess.run(
    [W,b,loss],{x:x_train,y:y_train})

print('W: %s b: %s loss: %s'%(curr_W,curr_b,curr_loss))
