import cv2, os, pickle, sys
import numpy as np
import tensorflow as tf
from utils.util import *
from tracking.tracker import Tracker
from datetime import datetime
import scipy.cluster.hierarchy as hcluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


currentUrl = os.path.dirname(__file__)
print('currentUrl', currentUrl)

# parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
# print('parentUrl', parentUrl)
sys.path.append(currentUrl)

from box2vec.resnet import Resnet
from box2vec.config import Config
def get_caps_and_pickles(video_files, detection_files):
    caps = []
    detections = []
    for video_file, detection_file in zip(video_files, detection_files):
        caps.append(cv2.VideoCapture(video_file))
        detections.append(np.load(detection_file))
    return caps, detections, len(detections[0])

def get_frames_and_boxes(caps, detections, index, show=True):
    frames = []
    local_bboxes = []
    for camera_id in range(len(caps)):
        cap = caps[camera_id]
        cap.set(1, index)
        ret, frame = cap.read()

        if not ret:
            raise

        frames.append(frame)
        
        local_bbox = detections[camera_id][index][:,:-1] # remove p
        local_bboxes.append(local_bbox)
        if show:
            for box in local_bbox:
                cv2.rectangle(
                            frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (255,0,0),
                            1)
            cv2.imshow('camera'+str(camera_id), frame)
            cv2.waitKey(1)
    return frames, local_bboxes

def get_embedding(local_bboxes, frames, box_to_vect, sess):
    padded_local_bboxes = []
    for camera_id, local_bbox in enumerate(local_bboxes):
        camera_id_pad = np.empty(shape=(local_bbox.shape[0], 1))
        camera_id_pad.fill(camera_id)
        local_bbox = np.concatenate((local_bbox, camera_id_pad), axis=1)
        padded_local_bboxes.append(local_bbox)
    if len(padded_local_bboxes) == 0:
        return
    bbox_batch = np.concatenate(padded_local_bboxes, axis=0)
    if bbox_batch.shape[0] == 0:
        return
    bbox_batch[:, 0] = bbox_batch[:, 0] / frames[0].shape[1] 
    bbox_batch[:, 2] = bbox_batch[:, 2] / frames[0].shape[1] 
    bbox_batch[:, 1] = bbox_batch[:, 1] / frames[0].shape[0] 
    bbox_batch[:, 3] = bbox_batch[:, 3] / frames[0].shape[0] 

    image_batch = []
    for frame in frames:
        image = cv2.resize(frame, (Config['image_width'] , Config['image_height']))
        # image = image[..., ::-1] # RGB -> BGR
        image_batch.append(image)
    image_batch = np.array(image_batch, dtype=np.uint8)
    box_ind_batch = bbox_batch[:, -1].astype(np.int32)
    # print('image_batch', image_batch.shape)
    # print('bbox_batch', bbox_batch.shape)
    # print('box_ind_batch', box_ind_batch.shape)
    embedding = box_to_vect.inference(image_batch, bbox_batch, box_ind_batch, sess)
    # print('embedding')
    # print(embedding)
    embedding = np.reshape(embedding, [-1, 128])
    return embedding

def init_box_to_vect_net(model_file):
    box_to_vect = Resnet(
                    image_height=Config['image_height'], 
                    image_width=Config['image_width'], 
                    vector_dim=128, 
                    alpha=Config['alpha'], 
                    feature_map_layer = 'block_layer3',
                    resnet_size=18,
                    data_format='channels_first', 
                    mode='test', 
                    # init_learning_rate=0.001,
                    # optimizer_name='adam',
                    # batch_size=Config['batch_size'],
                    # max_step=100000,
                    # model_path=r'F:\ubuntu\multi-camera-detection_v2\model',
                    # logdir=r'F:\ubuntu\multi-camera-detection_v2\log',
                    )

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_file)
    print('Load weights from', model_file, 'for box_to_vect!')
    return box_to_vect, sess

def l2_distance(embedding1, embedding2):

    extend_embedding1 = np.repeat(embedding1, embedding2.shape[0], axis = 0)
    extend_embedding2 = np.tile(embedding2, (embedding1.shape[0],1))
    distance = np.sqrt(np.sum(np.square(extend_embedding1 - extend_embedding2), axis=1))
    distance = distance.reshape([embedding1.shape[0], embedding2.shape[0]])
    return distance

def fusion(distance, distance_threshold):
    pairs = np.argwhere(distance < distance_threshold)
    cluster = Disjoint()
    for i in range(distance.shape[0]):
        cluster.create_set(i)

    for pair in pairs:
        cluster.merge_sets(pair[0], pair[1])
    return cluster

def fusion_cluster(bboxes, method='cluster', distance_threshold=0.1, nms_threshold=0.3):
    '''
        bboxes: bboxes to be fused
        method: 'cluster' or 'nms'
    '''
    if bboxes.shape[0]<=1:
        return Disjoint(), bboxes
    if method == 'cluster':
        clusters = hcluster.fclusterdata(bboxes, distance_threshold, criterion="distance", depth=2)
        print('clusters', clusters)
        cluster_num = len(set(clusters))
        cluster_set_data = [[] for _ in range(cluster_num)]
        for i, cluster_id in enumerate(clusters):
            cluster_set_data[cluster_id-1].append(i)

        cluster_count = np.zeros((cluster_num, 1), dtype=np.int32)
        cluster_set = Disjoint()
        cluster_set.sets = cluster_set_data
        fused_bboxes = np.zeros((cluster_num, bboxes.shape[1]))
        for k in range(bboxes.shape[0]):
           fused_bboxes[clusters[k]-1] += bboxes[k]
           cluster_count[clusters[k]-1] += 1
        fused_bboxes /= cluster_count
        # print(cluster_count)
        # print('fused_bboxes', fused_bboxes.shape)
        # print(np.where(cluster_count>1))
        # fused_bboxes = fused_bboxes[np.where(cluster_count>1)[0]]
        return cluster_set, fused_bboxes # (n, 4)
    elif method == 'nms':
        return nms(bboxes, nms_threshold)
    else:
        logging.error(method+'Not Implement yet')
        raise


def demo(save=True, use_cluster=True):
    file_names = [
        '4p-c0', '4p-c1', '4p-c2', '4p-c3'
    ]

    video_files = [
        os.path.join(currentUrl, 'data', 'train', 'lab', file_name+'.avi')\
            for file_name in file_names 
    ]

    detection_files = [
        os.path.join(currentUrl, 'data', 'train', 'lab', file_name+'.pickle')\
            for file_name in file_names
    ]

    model_file = os.path.join(currentUrl, 'box2vec', 'model', 'model-86000')
    cluster_distance_threshold = 0.7
    
    print('video_files:')
    print(video_files)
    print('detection_files:')
    print(detection_files)
    caps, detections, video_length = get_caps_and_pickles(video_files, detection_files)
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        now_str = str(datetime.now()).replace(':', '-')[:-7]
        cwd = os.getcwd()

        save_dir = os.path.join(cwd, 'result', now_str)
        os.mkdir(save_dir)
        video_names = [os.path.join(save_dir,'c_'+str(i)+'.avi') for i in range(len(caps))]
        outers = [cv2.VideoWriter(video_names[i], fourcc, 30.0, (360, 288)) for i in range(len(caps))]
    box_to_vect, sess = init_box_to_vect_net(model_file)
    tracker = Tracker(
                        traj_smoth_alpha = 0.99, 
                        image_boundary=None,
                        lost_times_thresh=30, 
                        lost_times_thresh_edge=3,
                        appear_times_thresh=30,
                        assoc_score_thresh = 0.7,
                        cost_metric='distance'
                        )

    for frame_id in range(100, video_length):
        print('-'*60)
        print('frame_id:', frame_id)
        frames, boxes = get_frames_and_boxes(caps, detections, frame_id, show=False)
        embedding = get_embedding(boxes, frames, box_to_vect, sess)
        if embedding is None:
            continue
        if embedding.shape[0] > 1:
            embedding_tsne = TSNE().fit_transform(embedding)
            plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1])
            plt.pause(1)
            plt.close('all')
        if use_cluster:
            cluster_set, fused_embeddings = fusion_cluster(bboxes=embedding, method='cluster', distance_threshold=1, nms_threshold=0.3)

        else:
            distance = l2_distance(embedding, embedding)
            print('distance:')
            print(distance)
            cluster_set = fusion(distance, distance_threshold=cluster_distance_threshold)
            print('cluster_sets:', len(cluster_set.sets))
            fused_embeddings = []
            for embedding_ids in cluster_set.sets:
                clustered_embedding = embedding[embedding_ids, :]
                fused_embeddings.append(np.mean(clustered_embedding, axis=0))
            fused_embeddings = np.concatenate(fused_embeddings, axis=0)
            fused_embeddings = np.reshape(fused_embeddings, [-1, 128])
            print('fused_embeddings:')
            print(fused_embeddings.shape)

        tracker.update_tracker(candidate_bboxes_original=fused_embeddings, time_stamp=frame_id)
        obj_to_traj = {}
        for valid_index in tracker.real_alive_index:
            print('valid_index:',valid_index)
            traj = tracker.trajectories[valid_index]
            current_object_id = traj.object_id[-1]
            traj_serial_num = tracker.whole_real_alive_index.index(valid_index)
            obj_to_traj[current_object_id] = traj_serial_num
            #print(traj_serial_num, ':', current_object_id)

        print(obj_to_traj)

        global_object_id_to_traj_id = [-1] * embedding.shape[0]
        for obj_id, cluster in enumerate(cluster_set.sets):
            for global_object_id in cluster:
                if obj_id in obj_to_traj:
                    global_object_id_to_traj_id[global_object_id] = obj_to_traj[obj_id]
        global_object_id = 0
        for camera_id in range(4):
            frame = frames[camera_id]
            local_bbox = boxes[camera_id]
            for box in local_bbox:
                traj_id = global_object_id_to_traj_id[global_object_id]
                # print(traj_id)
                np.random.seed(traj_id+10)
                color = np.random.randint(256, size=3)
                cv2.rectangle(
                            frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (int(color[0]), int(color[1]), int(color[2])),
                            1)
                
                global_object_id += 1
                cv2.putText(frame, str(traj_id), (box[0], box[3]),\
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), thickness = 2, lineType = -1)
            cv2.putText(frame, str(frame_id), (0, 20),\
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), thickness=2, lineType=-1)
            # print(frame.shape)
            if save:
                outers[camera_id].write(frame)
            cv2.imshow('camera'+str(camera_id), frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    demo(save=True)

