import numpy as np

class Disjoint():
    def __init__(self):
        self.sets = []

    def create_set(self, repr):
        self.sets.append([repr])

    def merge_sets(self, repr1, repr2):
        set1 = self.sets[self.find_set(repr1)];
        set2 = self.sets[self.find_set(repr2)];
        if set1 != set2:
            set1.extend(set2);
            self.sets.remove(set2);

    def find_set(self, repr1):
        for ind, one_set in enumerate(self.sets):
            if repr1 in one_set:
                return ind


    def get_sets(self):
        return self.sets

def union(au, bu, area_intersection):
    area_a = (au[:,2] - au[:, 0]) * (au[:, 3] - au[:, 1])
    area_b = (bu[:, 2] - bu[:, 0]) * (bu[:, 3] - bu[:, 1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = np.maximum(ai[:,0], bi[:,0])
    y = np.maximum(ai[:,1], bi[:,1])
    w = np.minimum(ai[:,2], bi[:,2]) - x
    h = np.minimum(ai[:,3], bi[:,3]) - y
    w[np.where(w < 0)] = 0
    h[np.where(h < 0)] = 0

    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    # get iou of boxes of a and b

    # if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
    #     return 0.0

    area_i = intersection(a, b)
    area_i[np.where(a[:,0] >= a[:,2])] = 0
    area_i[np.where(a[:,1] >= a[:,3])] = 0
    area_i[np.where(b[:,0] >= b[:,2])] = 0
    area_i[np.where(b[:,1] >= b[:,3])] = 0
    area_u = union(a, b, area_i)

    return area_i / (area_u + 1e-6)

def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=1))


def compute_cost_matrix(bbox_receiver, bbox_provider, metric='iou'):
    '''
        compute association cost matrix between bbox_receiver and bbox_provider
        metric: 'iou' or 'image_distance' or 'ground_distance'
    '''
    receiver_length = bbox_receiver.shape[0]
    provider_length = bbox_provider.shape[0]
    if metric == 'iou':
        extend_bbox_receiver = np.repeat(bbox_receiver, provider_length, axis = 0)
        extend_bbox_provider = np.tile(bbox_provider, (receiver_length,1))
        cost_matrix = iou(extend_bbox_receiver, extend_bbox_provider)
        cost_matrix = -cost_matrix.reshape([receiver_length,provider_length])
    elif metric == 'distance':
        receiver_center = (bbox_receiver[:,0:2] + bbox_receiver[:, 2:4]) / 2
        provider_center = (bbox_provider[:,0:2] + bbox_provider[:, 2:4]) / 2
        extend_bbox_receiver = np.repeat(receiver_center, provider_length, axis = 0)
        extend_bbox_provider = np.tile(provider_center, (receiver_length,1))
        cost_matrix = distance(extend_bbox_receiver, extend_bbox_provider)
        cost_matrix = cost_matrix.reshape([receiver_length,provider_length])
    else:
        raise
    return cost_matrix


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
