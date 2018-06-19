import tensorflow as tf 
import time
import os

class BoxToVect():
    def __init__(self, 
                # sess=None,
                mode='train',
                input_size=4,
                hiden_sizes=[16, 32, 32, 16], 
                output_size=2,
                logdir='/log',
                weights_decay=0.1,
                optimizer_name='sgd',
                init_learning_rate=0.001,
                max_step=100000,
                batch_size=256,
                dropout_keep=0.5,
                alpha=0.2,
                model_path='/model',
                ):
        '''
            Fully connected network
            Params:
                input_size: size of input
                hidden_sizes: hidden size and keep probability of dropout
        '''
        print('Initialize BoxToVect...')
        self.input_size = input_size
        self.hiden_sizes = hiden_sizes
        self.output_size = output_size
        self.mode = mode
        self.alpha = alpha

        # the two bboxes coordinates
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, input_size), name='input_bboxes') # [batch_size, 4]
        
        self.logdir = logdir
        self.optimizer_name = optimizer_name
        if self.mode == 'train':
            self.learning_rate = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.float32) # keep
        self.dorpout_keep = dropout_keep
        self.max_step = max_step
        self.model_path = model_path
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.weights_regularizer = tf.contrib.layers.l2_regularizer(weights_decay)

        self._build_network()
        print('Initialize Done...')
        # if not sess:
        #     self.sess = sess
        # else:
        #     self.sess = tf.Session()


    def _build_network(self):
        
        def _buid_net(x=self.x, reuse=False):
            with tf.name_scope('hidden_0'):
                hidden = tf.contrib.layers.fully_connected(
                        inputs=x,
                        num_outputs=self.hiden_sizes[0],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        normalizer_params=None,
                        # weights_initializer=initializers.xavier_initializer(),
                        weights_regularizer=self.weights_regularizer,
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        reuse=reuse,
                        variables_collections=None,
                        outputs_collections=None,
                        trainable=True,
                        scope='hidden_0'
                    )

                hidden = tf.contrib.layers.dropout(
                            inputs=hidden,
                            keep_prob=self.dropout,
                            noise_shape=None,
                            is_training=True,
                            outputs_collections=None,
                            scope='dropout'
                        )


            for layer_number in range(1, len(self.hiden_sizes)):
                with tf.name_scope('hidden_'+str(layer_number)):
                    hidden = tf.contrib.layers.fully_connected(
                            inputs=hidden,
                            num_outputs=self.hiden_sizes[layer_number],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            # weights_initializer=initializers.xavier_initializer(),
                            weights_regularizer=self.weights_regularizer,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=reuse,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope='hidden_'+str(layer_number)
                        )

                    hidden = tf.contrib.layers.dropout(
                            inputs=hidden,
                            keep_prob=self.dropout,
                            noise_shape=None,
                            is_training=True,
                            outputs_collections=None,
                            scope='dropout'
                        )


            with tf.name_scope('box_vector'):
                box_vector = tf.contrib.layers.fully_connected(
                            inputs=hidden,
                            num_outputs=self.output_size,
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            # weights_initializer=initializers.xavier_initializer(),
                            weights_regularizer=self.weights_regularizer,
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=reuse,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope='box_vector'
                        )
            return box_vector

        # for x1
        self.box_vector = _buid_net(x=self.x)
        self.embeddings = tf.nn.l2_normalize(self.box_vector, 1, 1e-10, name='embeddings')
        
        if self.mode == 'train':
            with tf.name_scope('losses'):
                anchor, positive, negative = tf.unstack(tf.reshape(self.embeddings, [-1, 3, self.output_size]), 3, 1)

                # self.similarity_loss = tf.reduce_mean(self.y * tf.reduce_mean(tf.square(self.box_vector2 - self.box_vector1), axis=1))
                self.similarity_loss = triplet_loss(anchor, positive, negative, self.alpha)
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.regularization_loss = tf.contrib.layers.apply_regularization(self.weights_regularizer, reg_variables)
                # self.regularization_loss = tf.losses.get_regularization_loss()

                self.loss = self.similarity_loss + self.regularization_loss

            similarity_loss_sum = tf.summary.scalar('similarity_loss', self.similarity_loss)
            reg_loss_sum = tf.summary.scalar('regularization_loss', self.regularization_loss)
            total_loss_sum = tf.summary.scalar('total_loss', self.loss)
            
            lr_sum = tf.summary.scalar('learning_rate', self.learning_rate)

            self.train_sum = tf.summary.merge(inputs=[similarity_loss_sum, reg_loss_sum, total_loss_sum, lr_sum])

            self.val_sum = tf.summary.merge(inputs=[similarity_loss_sum, total_loss_sum])

        tf.add_to_collection('match_placeholder_x_batch', self.x)
        tf.add_to_collection('match_placeholder_dropout', self.dropout)

    def _get_feed_dict(self, keep_dropout, x_batch, learning_rate=None):
        feed_dict = {self.x: x_batch,
                     self.dropout: keep_dropout}

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

    def inference(self, x_batch, sess):
        feed_dict = self._get_feed_dict(keep_dropout=1.0, x_batch=x_batch)
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

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
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
            x_train_batch = data_set.get_data_batch(batch_size=self.batch_size, split='train')
            # print('x_train_batch', x_train_batch.shape)
            learning_rate = self._get_learning_rate(global_step=step)
            feed_dict = self._get_feed_dict(
                                    keep_dropout=self.dorpout_keep, 
                                    x_batch=x_train_batch,
                                    learning_rate=learning_rate)
            get_data_time = time.time() - start_time_step
            # print('Get data batch time:', get_data_time)
            train_embeddings, _, sim_ls, reg_ls, tol_ls = sess.run([self.embeddings, train_op, self.similarity_loss, self.regularization_loss, self.loss], feed_dict)
            time_per_iter = time.time() - start_time_step

            
            if step % 100 == 0: # write summary
                print('Train:', step, 'total_ls:', tol_ls, 'sim_ls:', sim_ls, 'reg_ls:', reg_ls, 'time:', time_per_iter)
                
                
                summary = sess.run(self.train_sum, feed_dict)
                train_writer.add_summary(summary, step)
                
                # val set
                x_val_batch = data_set.get_data_batch(batch_size=self.batch_size, split='val')

                feed_dict = self._get_feed_dict(
                                    keep_dropout=1.0,
                                    x_batch=x_val_batch, 
                                    learning_rate=None)
                summary, val_embeddings, sim_ls, reg_ls, tol_ls = sess.run([self.val_sum, self.embeddings, self.similarity_loss, self.regularization_loss, self.loss],
                                                                         feed_dict=feed_dict)
                print('Val:', step, 'total_ls:', tol_ls, 'sim_ls:', sim_ls, 'reg_ls:', reg_ls, 'time:', time_per_iter)
                test_writer.add_summary(summary, step)
                print('train_embeddings:')
                print(train_embeddings[:10])
                print('-'*50)
                print('val_embeddings:')
                print(val_embeddings[:10])
                print('-'*50)


            if step % 10000 == 0: # save moel
                saver.save(sess, os.path.join(self.model_path, 'model'), global_step=step)
                print("model-%s saved." %(step))

        train_writer.close()
        test_writer.close()


def triplet_loss(anchor, positive, negative, alpha):
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

    