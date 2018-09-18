# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os
currentUrl = os.path.dirname(__file__)
print('currentUrl', currentUrl)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
print('parentUrl', parentUrl)
sys.path.append(parentUrl)
sys.path.append(currentUrl)
from config import Config
import time


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_FEATURE_SHAPE={
    18:{'block_layer4':(7,7,512), 'block_layer3':(14,14,256),
            'block_layer2':(28,28,128), 'block_layer1':(56,56,64)},

    34:{'block_layer4':(7,7,2048), 'block_layer3':(14,14,1024),
            'block_layer2':(28,28,512), 'block_layer1':(56,56,256)},

    50:{'block_layer4':(7,7,2048), 'block_layer3':(14,14,1024),
            'block_layer2':(28,28,512), 'block_layer1':(56,56,256)},

    101:{'block_layer4':(7,7,2048), 'block_layer3':(14,14,1024),
            'block_layer2':(28,28,512), 'block_layer1':(56,56,256)},

    152:{'block_layer4':(7,7,2048), 'block_layer3':(14,14,1024),
            'block_layer2':(28,28,512), 'block_layer1':(56,56,256)},

    200:{'block_layer4':(7,7,2048), 'block_layer3':(14,14,1024),
            'block_layer2':(28,28,512), 'block_layer1':(56,56,256)},
}
class Resnet(object):

    def __init__(self, 
            image_height=224, 
            image_width=224, 
            vector_dim=128, 
            alpha=0.5, 
            feature_map_layer = 'block_layer3',
            resnet_size=18,
            data_format='channels_first', 
            mode='train', 
            init_learning_rate=0.001,
            optimizer_name='sgd',
            batch_size=32,
            max_step=100000,
            model_path='model/',
            logdir='log/',
            ):
        '''
            Params:
                image: tensors or placeholder of shape[N, H, W, C]
        '''
        self.image = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3]) # shape[N,H,W,C]
        self.box = tf.placeholder(tf.float32, shape=[None, 5]) # shape [num_of_box, 5], within normalized [x1, y1, x2, y2, camera_id],
        self.box_ind = tf.placeholder(tf.int32, shape=[None,]) # A 1-D tensor of shape [num_boxes] with int32 values in [0, batch). 
                                                                # The value of box_ind[i] specifies the image that the i-th box refers to.

        self.resnet_size = resnet_size
        self.data_format = data_format
        self.feature_map_layer = feature_map_layer
        self.reuse_resnet = False
        self.vector_dim = vector_dim
        self.alpha = alpha
        self.mode = mode
        if self.mode == 'train':
            self.learning_rate = tf.placeholder(tf.float32)
        self.init_learning_rate = init_learning_rate
        self.max_step = max_step
        self.optimizer_name = optimizer_name
        self.mode = mode
        self.model_path = model_path
        self.logdir = logdir
        self.batch_size = batch_size
        self._build_net()
        print('Initialize Done...')
        # self.feature_shape = (224, 224, 128)
        print('resnet_size:', resnet_size)
        print('feature_map_layer:', feature_map_layer)
        print('alpha:', self.alpha)

    def _build_net(self):
        self.embeddings = self.feature_to_embedding()
        if self.mode == 'train':
            anchor, positive, negative = tf.unstack(tf.reshape(self.embeddings, [-1, 3, self.vector_dim]), 3, 1)
            self.loss = self.triplet_loss(anchor, positive, negative, self.alpha)
            loss_sum = tf.summary.scalar('loss', self.loss)
            # reg_loss_sum = tf.summary.scalar('regularization_loss', self.regularization_loss)
            # total_loss_sum = tf.summary.scalar('total_loss', self.loss)
            
            lr_sum = tf.summary.scalar('learning_rate', self.learning_rate)
            self.train_sum = tf.summary.merge(inputs=[loss_sum, lr_sum])

            self.val_sum = tf.summary.merge(inputs=[loss_sum])

    def _get_feed_dict(
                    self,
                    image_batch,
                    box_batch,
                    box_ind_batch,
                    learning_rate=None):
        feed_dict = {self.image: image_batch,
                     self.box: box_batch,
                     self.box_ind: box_ind_batch
                     }

        if learning_rate is not None:
            feed_dict[self.learning_rate] = learning_rate
        return feed_dict

    def _get_learning_rate(self, global_step):
        if global_step < 20000:
            return self.init_learning_rate
        if global_step < 40000:
            return self.init_learning_rate * 0.1
        if global_step < 80000:
            return self.init_learning_rate * 0.01
        else:
            return self.init_learning_rate * 0.01

    def inference(self, image_batch, box_batch, box_ind_batch, sess):
        feed_dict = self._get_feed_dict(
                    image_batch=image_batch,
                    box_batch=box_batch,
                    box_ind_batch=box_ind_batch,
                    learning_rate=None)
        embeddings = sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings

    def train(self, data_set, sess):
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(self.logdir+'/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(self.logdir + '/val')

        if self.optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=0.9
                )
        elif self.optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)  
        else:
            print(self.optimizer_name ,'optimizer noy supported!')
            raise

        train_op = optimizer.minimize(self.loss)
        print('='*50)
        start_step = 1
        tf.global_variables_initializer().run()
        try:
            ckpt_state = tf.train.get_checkpoint_state(self.model_path)
        except tf.errors.OutOfRangeError as e:
            print('Cannot restore checkpoint:', e)
            # tf.logging.error('Cannot restore checkpoint: %s', e)

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('No model yet at', self.model_path)
            # tf.logging.info('No model yet at %s', self.model_path)
        else:
            # tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            print('Loading checkpoint', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            start_step = int(ckpt_state.model_checkpoint_path.split('-')[-1])

        print('Start training ...')
        for step in range(start_step+1, self.max_step):
            start_time_step = time.time()

            image_batch, box_batch, box_ind_batch = data_set.get_data_batch(batch_size=self.batch_size, split='train')
            # print('x_train_batch', x_train_batch.shape)
            learning_rate = self._get_learning_rate(global_step=step)
            feed_dict = self._get_feed_dict(
                                    image_batch=image_batch,
                                    box_batch=box_batch,
                                    box_ind_batch=box_ind_batch,
                                    learning_rate=learning_rate
                                    )
            get_data_time = time.time() - start_time_step

            train_embeddings, _, ls = sess.run([self.embeddings, train_op, self.loss], feed_dict)

            time_per_iter = time.time() - start_time_step

            if step % 1 == 0: # write summary
                
                summary = sess.run(self.train_sum, feed_dict)
                train_writer.add_summary(summary, step)
                
                # val set
                image_batch_val, box_batch_val, box_ind_batch_val = data_set.get_data_batch(
                            batch_size=self.batch_size, 
                            split='val')

                feed_dict_val = self._get_feed_dict(
                                    image_batch=image_batch_val,
                                    box_batch=box_batch_val,
                                    box_ind_batch=box_ind_batch_val,
                                    learning_rate=None
                                    )
                summary_val, val_embeddings, ls_val = sess.run([
                                                    self.val_sum, 
                                                    self.embeddings, self.loss],
                                                    feed_dict=feed_dict_val)

                test_writer.add_summary(summary_val, step)
                print('Step:', step, 'train_loss:', ls, 'val_loss:', ls_val, 'time:', time_per_iter)
                #print('train_embeddings:')
                #print(train_embeddings[:10])
                #print('-'*50)
                #print('val_embeddings:')
                #print(val_embeddings[:10])
                #print('-'*50)


            if step % Config['save_every_step'] == 0: # save moel
                saver.save(sess, os.path.join(self.model_path, 'model'), global_step=step)
                print("model-%s saved." %(step))

        train_writer.close()
        test_writer.close()

    def triplet_loss(self, anchor, positive, negative, alpha):
        """Calculate the triplet loss according to the FaceNet paper
        
        Args:
          anchor: the embeddings for the anchor images.
          positive: the embeddings for the positive images.
          negative: the embeddings for the negative images.
      
        Returns:
          the triplet loss according to the FaceNet paper as a float tensor.
        """
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
            neg_dist = -tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
            
            basic_loss = tf.add(tf.add(pos_dist,neg_dist), alpha)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
          
        return loss

    def batch_norm_relu(self, inputs, is_training, data_format):
        """Performs a batch normalization followed by a ReLU."""
        # We set fused=True for a significant performance boost.
        # See https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, fused=True)
            # inputs = tf.where(inputs > 0, inputs, tf.nn.sigmoid(inputs) - 0.5)
        inputs = tf.nn.relu(inputs)
        return inputs

    def fixed_padding(self, inputs, kernel_size, data_format):
        """Pads the input along the spatial dimensions independently of input size.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
            kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                     Should be a positive integer.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            A tensor with the same format as the input with the data either intact
            (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format):
        """Strided 2-D convolution with explicit padding.

        The padding is consistent and is based only on `kernel_size`, not on the
        dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        """
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)

        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def conv2d_transpose(self, inputs, filters, kernel_size, strides, data_format):
        return tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding='same', data_format=data_format, 
            kernel_initializer=tf.variance_scaling_initializer()
            )

    def building_block(self,inputs, filters, is_training, projection_shortcut, strides,
                       data_format):
        """Standard building block for residual networks with BN before convolutions.

        Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        is_training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts (typically
            a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block.
        """
        shortcut = inputs
        inputs = self.batch_norm_relu(inputs, is_training, data_format)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

        inputs = self.batch_norm_relu(inputs, is_training, data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)

        return inputs + shortcut


    def bottleneck_block(self, inputs, filters, is_training, projection_shortcut,
                         strides, data_format):
        """Bottleneck block variant for residual networks with BN before convolutions.

        Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first two convolutions. Note that the
          third and final convolution will use 4 times as many filters.
        is_training: A Boolean for whether the model is in training or inference
          mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts (typically
          a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
          downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block.
        """
        shortcut = inputs
        inputs = self.batch_norm_relu(inputs, is_training, data_format)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)

        inputs = self.batch_norm_relu(inputs, is_training, data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

        inputs = self.batch_norm_relu(inputs, is_training, data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)
        
        return inputs + shortcut


    def block_layer(self, inputs, filters, block_fn, blocks, strides, is_training, name,
                    data_format):
        """Creates one layer of blocks for the ResNet model.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the first convolution of the layer.
                block_fn: The block to use within the model, either `building_block` or
                `bottleneck_block`.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
                greater than 1, this layer will ultimately downsample the input.
            is_training: Either True or False, whether we are currently training the
                model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block layer.
        """
        # Bottleneck blocks end with 4x the number of filters as they start with
        if block_fn == 'building_block':
            block_fn = self.building_block
            filters_out = filters
        else:
            block_fn = self.bottleneck_block
            filters_out = 4 * filters
        # filters_out = 4 * filters if block_fn is self.bottleneck_block else filters
        # filters_out = 4 * filters
        def projection_shortcut(inputs):
            return self.conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                data_format=data_format)

        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                        data_format)

        for i in range(1, blocks):
            inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

        return tf.identity(inputs, name)


    def features_v2_generator(self, block_fn, layers, is_training=True, feature_map_layer='block_layer4',
                                    data_format=None):
        """Generator for ResNet v2 models.

        Args:
            block_fn: The block to use within the model, either `building_block` or
                `bottleneck_block`.
            layers: A length-4 array denoting the number of blocks to include in each
                layer. Each layer consists of blocks that take inputs of the same size.
            num_classes: The number of possible classes for image classification.
            data_format: The input format ('channels_last', 'channels_first', or None).
                If set to None, the format is dependent on whether a GPU is available.

          Returns:
            The model function that takes in `inputs` and `is_training` and
            returns the output tensor of the ResNet model.
          """

        print('Get feature map:',feature_map_layer, 'with is_training=', is_training, 'reuse:', self.reuse_resnet)
        if data_format is None:
            data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

        if data_format == 'channels_first':
            # Convert from channels_last (NHWC) to channels_first (NCHW). This
            # provides a large performance boost on GPU.
            
            inputs = tf.transpose(self.image, [0, 3, 1, 2]) # image_batch: [N, C, H, W]
        else:
            inputs = tf.identity(self.image)
        
        with tf.variable_scope('resnet',reuse=self.reuse_resnet):
            self.reuse_resnet = True
            print('image_size:', inputs.shape)
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=64, kernel_size=7, strides=1,
                data_format=data_format)
            print('conv_1:', inputs.shape)
            inputs = tf.identity(inputs, 'initial_conv')
            print('identity_1:', inputs.shape)
            # inputs = tf.layers.max_pooling2d(
            #     inputs=inputs, pool_size=3, strides=2, padding='SAME',
            #     data_format=data_format)
            # print('pool_1:', inputs.shape)

            inputs = tf.identity(inputs, 'initial_max_pool')
            print('identity_2:', inputs.shape)
            inputs = self.block_layer(
                inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
                strides=2, is_training=is_training, name='block_layer1',
                data_format=data_format)
            blk_1 = inputs
            print('block_layer_1:', inputs.shape)
            if feature_map_layer == 'block_layer1':
                inputs = self.batch_norm_relu(inputs, is_training, data_format)
                if data_format == 'channels_first':
                    inputs = tf.transpose(inputs, [0, 2, 3, 1]) #convert back to (NHWC)
                print('feature map shape:', inputs.shape)
                return inputs


            inputs = self.block_layer(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
                strides=2, is_training=is_training, name='block_layer2',
                data_format=data_format)
            blk_2 = inputs
            print('block_layer_2:', inputs.shape)

            if feature_map_layer == 'block_layer2':
                inputs = self.batch_norm_relu(inputs, is_training, data_format)
                if data_format == 'channels_first':
                    inputs = tf.transpose(inputs, [0, 2, 3, 1]) #convert back to (NHWC)
                print('feature map shape:', inputs.shape)
                return inputs

            inputs = self.block_layer(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
                strides=1, is_training=is_training, name='block_layer3',
                data_format=data_format)
            print('block_layer_3:', inputs.shape)

            if feature_map_layer == 'block_layer3':
                inputs = self.batch_norm_relu(inputs, is_training, data_format)
                if data_format == 'channels_first':
                    inputs = tf.transpose(inputs, [0, 2, 3, 1]) #convert back to (NHWC)
                print('feature map shape:', inputs.shape)
                return inputs

            inputs = self.block_layer(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
                strides=1, is_training=is_training, name='block_layer4',
                data_format=data_format)
            print('block_layer_4:', inputs.shape)
            if feature_map_layer == 'block_layer4':
                inputs = self.batch_norm_relu(inputs, is_training, data_format)
                if data_format == 'channels_first':
                    inputs = tf.transpose(inputs, [0, 2, 3, 1]) #convert back to (NHWC)
                print('feature map shape:', inputs.shape)
                return inputs

            # inputs = batch_norm_relu(inputs, is_training, data_format)
            # inputs = tf.layers.average_pooling2d(
            #     inputs=inputs, pool_size=7, strides=1, padding='VALID',
            #     data_format=data_format)
            # inputs = tf.identity(inputs, 'final_avg_pool')
            # inputs = tf.reshape(inputs, [inputs.get_shape()[0].value, -1])
            # inputs = tf.layers.dense(inputs=inputs, units=num_classes)
            # inputs = tf.identity(inputs, 'final_dense')
            print('feature map shape:', inputs.shape)
            return inputs     

    def resnet_v2_maps(self, is_training=True):
        """Returns the ResNet feature maps given layer, shape[N,H,W,C]"""
        model_params = {
            18: {'block': 'building_block', 'layers': [2, 2, 2, 2]},
            34: {'block': 'building_block', 'layers': [3, 4, 6, 3]},
            50: {'block': 'bottleneck_block', 'layers': [3, 4, 6, 3]},
            101: {'block': 'bottleneck_block', 'layers': [3, 4, 23, 3]},
            152: {'block': 'bottleneck_block', 'layers': [3, 8, 36, 3]},
            200: {'block': 'bottleneck_block', 'layers': [3, 24, 36, 3]}
        }

        if self.resnet_size not in model_params:
            raise ValueError('Not a valid resnet_size:', self.resnet_size)

        params = model_params[self.resnet_size]
        features_map = self.features_v2_generator(
            params['block'], params['layers'], feature_map_layer=self.feature_map_layer, is_training=is_training, data_format=self.data_format)
        
        return features_map

    def feature_to_embedding(self):
        feature_map = self.resnet_v2_maps() # [image_batch, h, w, c]
        x1 = tf.reshape(self.box[:, 0], [-1,1])
        y1 = tf.reshape(self.box[:, 1], [-1,1])
        x2 = tf.reshape(self.box[:, 2], [-1,1])
        y2 = tf.reshape(self.box[:, 3], [-1,1])
        boxes = tf.concat([y1, x1, y2, x2], axis=1)

        # shape [num_boxes, crop_height, crop_width, depth]
        roi_feature_map = tf.image.crop_and_resize(
            image = feature_map,
            boxes = boxes,
            box_ind = self.box_ind,
            crop_size = Config['roi_pooling_size']
            )

        flatten_roi_feature_map = tf.contrib.layers.flatten(roi_feature_map) # [num_boxes, ...]
        feature_with_position = tf.concat([flatten_roi_feature_map, self.box], axis=1)
        print('feature_with_position:', feature_with_position.shape)
        vetctor = tf.contrib.layers.fully_connected(
                        inputs=feature_with_position,
                        num_outputs=self.vector_dim,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        # weights_initializer=initializers.xavier_initializer(),
                        # weights_regularizer=self.weights_regularizer,
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        # reuse=reuse,
                        variables_collections=None,
                        outputs_collections=None,
                        trainable=True
                    )
        embeddings = tf.nn.l2_normalize(vetctor, 1, 1e-10, name='embeddings')
        return embeddings
        # anchor, positive, negative = tf.unstack(tf.reshape(self.embeddings, [-1, 3, self.vector_dim]), 3, 1)




def resize_image(image, new_size=(224,224)):
    '''
    crop to a square with length of shorter size of image and resize to (224, 224)
    '''
    width, height = image.size
    if width > height:
        left = (width - height) // 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([new_size[0], new_size[1]], Image.ANTIALIAS)
    return image



if __name__ == '__main__':
    
    with open('1.jpg', 'rb') as f:
        with Image.open(f) as image:
                image = np.asarray(resize_image(image, (Config['image_width'], Config['image_height'])))
                image = np.expand_dims(image, axis=0)
    # images = tf.placeholder(tf.float32, shape=image.shape)
    resnet = Resnet(image_height=Config['image_height'], image_width=Config['image_width'], 
        feature_map_layer = 'block_layer4',
        resnet_size=50, data_format='channels_first')
    feed_dict = {resnet.image: image}

    feature_maps = resnet.resnet_v2_maps()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        feats = sess.run(feature_maps, feed_dict=feed_dict)
        print(feats.shape)
