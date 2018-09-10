# this is more of a playground to evaluate tensorflow expressions

# -*- coding: utf-8 -*-
import tensorflow as tf

num_anchors = 3
grid_size = [13, 13]

''' create x_offset '''
# do the first concat as we have to increment grid_line_to_add
grid_temp = tf.zeros([1, grid_size[0]], dtype=tf.float32)
grid_line_to_add = tf.ones([1, grid_size[0]], dtype=tf.float32)
grid_temp = tf.concat([grid_temp, grid_line_to_add], 0)

# assumption is made that grid_size is symetrical etc [13, 13]
for i in range(2, grid_size[0]): # start at 2 as we already concatenated once
  # increment by 1
  grid_line_to_add = tf.add(grid_line_to_add, tf.ones([1, grid_size[0]], dtype=tf.float32))
  grid_temp = tf.concat([grid_temp, grid_line_to_add], 0)

y_offset = tf.reshape(grid_temp, (-1, 1))
''' create y_offset end '''
'''create x_offset'''
x_offset = tf.constant([i for i in range(grid_size[0])], dtype=tf.float32)
x_offset_single = x_offset

for i in range(1, grid_size[0]): # start at 1 as we already have the first x_offset
  x_offset = tf.concat([x_offset, x_offset_single], 0)

x_offset = tf.reshape(x_offset, (-1, 1))

x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
x_y_offset_single = x_y_offset
''' create x_offset end '''

# print x_y_offset
temp = x_y_offset #####################


for i in range(num_anchors-1):
  x_y_offset = tf.concat([x_y_offset, x_y_offset_single], 1)

x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
init = tf.global_variables_initializer()


with tf.Session() as sess:
  sess.run(init)
  test = sess.run(temp)
  
  print test