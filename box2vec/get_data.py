import numpy as np 
import cv2
# TRAIN_SET_FILE = 'train.npy'
# VAL_SET_FILE = 'val.npy'
DATA_SIZE = -1
class BoxToVectDataSet():
    def __init__(self, train_set_file, val_set_file, video_files=[], test_set_file=None, image_width=224, image_height=224):
        
        self.data_set = {
                        'train': np.load(train_set_file)[:DATA_SIZE],
                        'val': np.load(val_set_file)[:DATA_SIZE],
                        'caps': [cv2.VideoCapture(video_file) for video_file in video_files]
        }

        if test_set_file:
            self.data_set['test'] = np.load(test_set_file)[:DATA_SIZE]


        self.data_size = {
                        'train':self.data_set['train'].shape[0],
                        'val':self.data_set['val'].shape[0],
                        'caps': self.data_set['caps'][0].get(7) # video length of one cap
        }

        if test_set_file:
            self.data_size['test'] = self.data_set['test'].shape[0]


        print('Train samples size:', self.data_size['train'])
        print('Val samples size:', self.data_size['val'])

        if test_set_file:
            print('Test samples size:', self.data_set['test'].shape)

        self.image_width = image_width
        self.image_height = image_height

        
    def get_data_batch(self, batch_size, split):
        '''
            batch_size: number of triplets in batch
        '''
        indexes = np.random.choice(self.data_size[split], batch_size)
        triplet_batch = self.data_set[split][indexes] # [batch_size, 3, 6]
        triplet_batch = np.reshape(triplet_batch, (-1, triplet_batch.shape[-1])) # [3*batch_size, 6] = [x1, y1, x2, y2, camera_id, frame_id]
        #print('triplet_batch', triplet_batch.shape)
        box_ind_batch = []
        image_batch = []
        global_frame_id_to_box_ind = {}
        
        for camera_id, frame_id in zip(triplet_batch[:, 4], triplet_batch[:, 5]):
            camera_id, frame_id = int(camera_id), int(frame_id)
            global_frame_id = int(self.data_size['caps'] * camera_id + frame_id)
            #print(camera_id, frame_id, global_frame_id)
            if global_frame_id not in global_frame_id_to_box_ind:
                global_frame_id_to_box_ind[global_frame_id] = len(image_batch)
                self.data_set['caps'][camera_id].set(1, frame_id)
                ret, frame = self.data_set['caps'][camera_id].read()
                frame = cv2.resize(frame, (self.image_width , self.image_height))
                # cv2.imshow('img', frame)
                # cv2.waitKey(1)
                image_batch.append(frame)
            box_ind_batch.append(global_frame_id_to_box_ind[global_frame_id])
        # print(triplet_batch.shape)
        return np.array(image_batch, dtype=np.uint8), triplet_batch[:, :-1], np.array(box_ind_batch, dtype=np.int32)

def denormalize_bboxes(bboxes, image_width, image_height):
    bboxes[np.where(bboxes > 1.0)] = 1.0
    bboxes[np.where(bboxes < 0.)] = 0.
    bboxes[:,0::2] *= image_width # x1,x2
    bboxes[:,1::2] *= image_height # y1, y2

    return bboxes

if __name__ == '__main__':
    video_dir = '/media/mhttx/F/project_developing/multi-camera-detection/data/train/lab/'
    dataset = BoxToVectDataSet(
            train_set_file='data/train.npy', 
            val_set_file='data/val.npy', 
            video_files=[
                video_dir+'4p-c0.avi',
                video_dir+'4p-c1.avi',
                video_dir+'4p-c2.avi',
                video_dir+'4p-c3.avi'
            ]
        )
    image_batch, box_batch, box_ind_batch = dataset.get_data_batch(4,'train')
    #print('image_batch\n', image_batch.shape)
    #print('box_batch\n', box_batch.shape)
    #print('box_ind_batch\n', box_ind_batch)
