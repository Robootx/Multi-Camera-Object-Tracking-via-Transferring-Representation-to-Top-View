import numpy as np 
import os
from get_data import BoxToVectDataSet
# from box2vect import BoxToVect
import tensorflow as tf 
from config import Config
from resnet import Resnet
import time
currentUrl = os.path.dirname(__file__)
print('currentUrl', currentUrl)

parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
print('parentUrl', parentUrl)

def train():
    video_dir = os.path.join(parentUrl, 'data', 'train', 'lab')
    box_to_vect_dataset = BoxToVectDataSet(
            train_set_file='data/train.npy', 
            val_set_file='data/val.npy', 
            video_files=[
                os.path.join(video_dir, '4p-c0.avi'),
                os.path.join(video_dir, '4p-c1.avi'),
                os.path.join(video_dir, '4p-c2.avi'),
                os.path.join(video_dir, '4p-c3.avi'),
            ],
            test_set_file=None, 
            image_width=Config['image_width'], 
            image_height=Config['image_height']
        )
    box_to_vect = Resnet(
            image_height=Config['image_height'], 
            image_width=Config['image_width'], 
            vector_dim=128, 
            alpha=Config['alpha'], 
            feature_map_layer = 'block_layer3',
            resnet_size=18,
            data_format='channels_first', 
            mode='train', 
            init_learning_rate=0.001,
            optimizer_name='adam',
            batch_size=Config['batch_size'],
            max_step=Config['max_step'],
            model_path='model/',
            logdir='log/',
            )

    with tf.Session() as sess:
        box_to_vect.train(box_to_vect_dataset, sess)

def test(model_path='model/model-30000'):
    video_dir = os.path.join(parentUrl, 'data', 'train', 'lab')
    box_to_vect_dataset = BoxToVectDataSet(
            train_set_file='data/train.npy', 
            val_set_file='data/val.npy', 
            video_files=[
                os.path.join(video_dir, '4p-c0.avi'),
                os.path.join(video_dir, '4p-c1.avi'),
                os.path.join(video_dir, '4p-c2.avi'),
                os.path.join(video_dir, '4p-c3.avi'),
            ],
            test_set_file='data/test.npy', 
            image_width=Config['image_width'], 
            image_height=Config['image_height']
        )
    box_to_vect = Resnet(
            image_height=Config['image_height'], 
            image_width=Config['image_width'], 
            vector_dim=128, 
            alpha=Config['alpha'], 
            feature_map_layer = 'block_layer3',
            resnet_size=18,
            data_format='channels_first', 
            mode='test', 
            init_learning_rate=0.001,
            optimizer_name='adam',
            batch_size=Config['batch_size'],
            model_path='model/',
            logdir='log/',
            )

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print('Load weights form', model_path)
    max_ap = 0
    min_an = 100
    for _ in range(100):

        image_batch, box_batch, box_ind_batch = box_to_vect_dataset.get_data_batch(batch_size=Config['batch_size'], split='test')
        
        embedding = box_to_vect.inference(image_batch, box_batch, box_ind_batch, sess)
        embedding = np.reshape(embedding, [-1, 3, embedding.shape[-1]])
        anchor, positive, negative = embedding[:, 0, :], embedding[:, 1, :], embedding[:, 2, :]
        # print('anchor\n', anchor)
        # print('positive\n', positive)
        # print('negative\n', negative)

        a_p_squared = np.sqrt(np.sum(np.square(anchor-positive), axis=1))
        a_n_squared = np.sqrt(np.sum(np.square(anchor-negative), axis=1))
        print('-'*50)
        # print('a_p_squared:\n', a_p_squared)
        # print('a_n_squared:\n', a_n_squared)
        a_p_squared_max = np.max(a_p_squared)
        a_n_squared_min = np.min(a_n_squared)
        print('a_p_squared_max:', a_p_squared_max)
        print('a_n_squared_min:', a_n_squared_min)
        # print('diff:\n', a_p_squared-a_n_squared)
        if max_ap < a_p_squared_max:
            max_ap = a_p_squared_max
        if min_an > a_n_squared_min:
            min_an = a_n_squared_min
        print('max_ap', max_ap)
        print('min_an', min_an)


if __name__ == '__main__':
    train()
    # test()