import pickle
import numpy as np
from random import shuffle
def get_triplet(dataset_file, image_width=None, image_height=None):
    '''
        record: 
    '''
    apns = []
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
        record, detections = dataset['record'], dataset['bbox']
    total_frame = len(record.keys())
    for i, frame_id in enumerate(record.keys()):
        print('Processing frame', i+1, '/', total_frame)
        record_of_cameras = record[frame_id]
        camera_ids = record_of_cameras.keys()
        for anchor_camera_id in camera_ids:
            for anchor_obj_id in record_of_cameras[anchor_camera_id]['obj2box']:
                anchor_box_id = record_of_cameras[anchor_camera_id]['obj2box'][anchor_obj_id]
                anchor = detections[anchor_camera_id][frame_id][anchor_box_id][:-1]
                anchor = np.concatenate((anchor, [anchor_camera_id, frame_id]))
                for positive_camera_id in camera_ids:
                    if positive_camera_id != anchor_camera_id and \
                        anchor_obj_id in record_of_cameras[positive_camera_id]['obj2box']:
                        positive_box_id = record_of_cameras[positive_camera_id]['obj2box'][anchor_obj_id]
                        positive = detections[positive_camera_id][frame_id][positive_box_id][:-1]
                        positive = np.concatenate((positive, [positive_camera_id, frame_id]))
                        for negative_camera_id in camera_ids:
                            for negative_obj_id in record_of_cameras[negative_camera_id]['obj2box']:
                                if negative_obj_id != anchor_obj_id:
                                    negative_box_id = record_of_cameras[negative_camera_id]['obj2box'][negative_obj_id]
                                    negative = detections[negative_camera_id][frame_id][negative_box_id][:-1]
                                    negative = np.concatenate((negative, [negative_camera_id, frame_id]))
                                    apn = np.concatenate([[anchor], [positive], [negative]])
                                    apns.append(apn)

    shuffle(apns)
    apns = np.array(apns)
    print(apns.shape)
    apns[:, :, 0] = apns[:, :, 0] / image_width
    apns[:, :, 2] = apns[:, :, 2] / image_width
    apns[:, :, 1] = apns[:, :, 1] / image_height
    apns[:, :, 3] = apns[:, :, 3] / image_height
    
    print('Total triplet:', len(apns))
    print('Triplet shape:', apns.shape)

    train_size = int((len(apns)) * 0.8)
    val_size = int((len(apns)) * 0.1)
    # test_size = int((len(apns) / 3) - train_size - val_size)

    tain_apns = apns[:train_size]
    val_apns = apns[train_size:(train_size+val_size)]
    test_apns = apns[(train_size+val_size):]
    print('train_size', len(tain_apns))
    print('val_size', len(val_apns))
    print('test_size', len(test_apns))
    np.save('train.npy', tain_apns)
    np.save('val.npy', val_apns)
    np.save('test.npy', test_apns)                                

if __name__ == '__main__':
    get_triplet(
        dataset_file='/media/mhttx/F/project_developing/multi-camera-detection/data/train/lab/annotations/dataset_3000_0106.pickle',
        image_width=360, 
        image_height=288)  
