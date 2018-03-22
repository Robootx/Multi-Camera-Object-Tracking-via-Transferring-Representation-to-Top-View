import numpy as np
import os, cv2, time, pickle, sys
import math

# global variables
caps = []
detections = []
camera_num = 0
video_length = 0
images_grid_cols = 2
images_grid_rows = 0
image_width = 0
image_height = 0
frame_num = 0
unioned_frame = None
def initialize(video_files, detection_files):

    global caps, detections, camera_num, video_length, unioned_frame
    global images_grid_cols, images_grid_rows, image_width, image_height

    camera_num = len(video_files)
    for video_file, detection_file in zip(video_files, detection_files):
        caps.append(cv2.VideoCapture(video_file))
        detections.append(np.load(detection_file))

    # check
    for i in range(camera_num):
        tmp_video_length = int(caps[i].get(7))
        if i == 0:
            video_length = tmp_video_length
        assert video_length == tmp_video_length
        assert len(detections[i]) == tmp_video_length

    image_width = int(caps[0].get(3))
    image_height = int(caps[0].get(4))

    images_grid_rows = int(math.ceil(camera_num / images_grid_cols))
    unioned_frame = 255 * np.ones((image_height*images_grid_rows,image_width*images_grid_cols,3), np.uint8)

    # for frame_num in range(video_length):

    #     frames = []
    #     local_bboxes = []
    #     global_bboxes = []
    #     for i in range(camera_num):
    #         ret, frame = caps[i].read()
    #         if not ret:
    #             raise
    #         row_id = i // images_grid_cols
    #         col_id = i % images_grid_cols
    #         x_bias = image_width * col_id
    #         y_bias = image_height * row_id
    #         frames.append(frame)
    #         local_bbox = detections[i][frame_num][:,:-1] # remove p
    #         global_bbox = np.zeros(local_bbox.shape, dtype='int32')
    #         global_bbox[:,0::2] = local_bbox[:,0::2] + x_bias
    #         global_bbox[:,1::2] = local_bbox[:,1::2] + y_bias
    #         global_bboxes.append(global_bbox)
    #         local_bboxes.append(local_bbox)

    #     unioned_frame = union_frames(
    #                 frames, 
    #                 frames_grid_rows=images_grid_rows, 
    #                 frames_grid_cols=images_grid_cols, 
    #                 split_line_width=2) 
    #     for global_bbox in global_bboxes:
    #         for box in global_bbox: # (x1, y1, x2, y2, p)
    #                 cv2.rectangle(
    #                         unioned_frame, 
    #                         (box[0], box[1]), 
    #                         (box[2], box[3]), 
    #                         (0,255,0),
    #                         1)
    #     cv2.imshow('Img', unioned_frame)
    #     cv2.waitKey(20)

def position_in_bbox(x, y, bbox):
    return x>=bbox[0] and x<=bbox[2] and y>=bbox[1] and y<=bbox[3]

def get_bbox_from_position(x, y, global_bboxes):
    camera_id_and_bbox_id = []
    for camera_id in range(len(global_bboxes)):
        for bbox_id in range(global_bboxes[camera_id].shape[0]):
            if position_in_bbox(x, y, global_bboxes[camera_id][bbox_id]):
                camera_id_and_bbox_id.append((camera_id, bbox_id))
    return camera_id_and_bbox_id

def call_back(event, x, y, flags, param):
    global caps, detections, frame_num, camera_num, unioned_frame
    print('event', event)
    print('flags', flags)
    # right button down, load next frames
    if event == cv2.EVENT_MOUSEWHEEL and flags > 0:
        print(event)
        print('frame_num:', frame_num)
        frames = []
        local_bboxes = []
        global_bboxes = []
        for i in range(camera_num):
            ret, frame = caps[i].read()
            if not ret:
                raise
            row_id = i // images_grid_cols
            col_id = i % images_grid_cols
            x_bias = image_width * col_id
            y_bias = image_height * row_id
            frames.append(frame)
            local_bbox = detections[i][frame_num][:,:-1] # remove p
            global_bbox = np.zeros(local_bbox.shape, dtype='int32')
            global_bbox[:,0::2] = local_bbox[:,0::2] + x_bias
            global_bbox[:,1::2] = local_bbox[:,1::2] + y_bias
            global_bboxes.append(global_bbox)
            local_bboxes.append(local_bbox)

        unioned_frame = union_frames(
                    frames, 
                    frames_grid_rows=images_grid_rows, 
                    frames_grid_cols=images_grid_cols, 
                    split_line_width=2) 
        for global_bbox in global_bboxes:
            for box in global_bbox: # (x1, y1, x2, y2, p)
                    cv2.rectangle(
                            unioned_frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (0,255,0),
                            1)
        
        frame_num += 1


        


def union_frames(frames, frames_grid_rows=2, frames_grid_cols=2, split_line_width=0):
    single_image_height, single_image_width = frames[0].shape[0], frames[0].shape[1]
    union_frame = np.zeros((single_image_height*frames_grid_rows, single_image_width*frames_grid_cols, 3), dtype=frames[0].dtype)
    for i in range(len(frames)):
        row_id = i // frames_grid_cols
        col_id = i % frames_grid_cols
        union_frame[row_id*single_image_height:row_id*single_image_height+single_image_height,\
                      col_id*single_image_width:col_id*single_image_width+single_image_width, :]=\
                      frames[i]
    for i in range(1, frames_grid_rows):
        # union_frame[:, (col_id*single_image_width-split_line_width):(col_id*single_image_width+split_line_width)] = 255
        union_frame[(i*single_image_height-split_line_width):(i*single_image_height+split_line_width),:] = 255
    for j in range(1, frames_grid_cols):
        union_frame[:, (j*single_image_width-split_line_width):(j*single_image_width+split_line_width)] = 255
    return union_frame


if __name__ == '__main__':
    root_dir = '/media/mhttx/F/ubuntu/multi-camera-detection/data/train/lab'
    video_files = ['4p-c0.avi', '4p-c1.avi', '4p-c2.avi', '4p-c3.avi']
    video_files = [os.path.join(root_dir, video_file) for video_file in video_files]
    detection_files = ['4p-c0.pickle', '4p-c1.pickle', '4p-c2.pickle', '4p-c3.pickle']
    detection_files = [os.path.join(root_dir, detection_file) for detection_file in detection_files]
    initialize(video_files, detection_files)

    cv2.namedWindow('Img', )
    cv2.setMouseCallback('Img',call_back)
    while True:
        cv2.imshow('Img', unioned_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break