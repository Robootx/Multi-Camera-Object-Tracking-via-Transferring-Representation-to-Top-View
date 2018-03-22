import numpy as np
from tracking.utils import *
from scipy.optimize import linear_sum_assignment
from collections import deque

class Trajectory():
    def __init__(self, strat_bbox, start_object_id, born_time, image_boundary, smoth_alpha=0.0):
        self.born_time = born_time      # time when trajectory generated
        self.is_alive = True            # target disappear or not
        self.bboxes = deque(maxlen=100)                # bboxes contained by the trajectory,a numpy array with shape [1,4]
        self.object_id = []
        self.smothed_bboxes = deque(maxlen=100)        # smothed_points
        self.smoth_alpha = smoth_alpha
        self.lost_times = 0             # times bbox disappeared continuesly
        self.average_speed = 0          # exponentional or moving speed(in ground plane or image)
        self.image_boundary = image_boundary
        self.appear_time = 0
        self.add_node(strat_bbox, start_object_id)

    def add_smothed_bbox(self):
        # assert len(self.bboxes) == len(self.smothed_bboxes) + 1, 'Add smothed bbox not allowed!'

        if len(self.smothed_bboxes) == 0:
            self.smothed_bboxes.append(self.bboxes[-1])
        else:
            smoth_to_add = self.smoth_alpha * self.bboxes[-1] \
                        + (1-self.smoth_alpha) * self.smothed_bboxes[-1] 
            self.smothed_bboxes.append(smoth_to_add)

    def add_node(self, raw_bbox, object_id):
        self.bboxes.append(raw_bbox)
        self.object_id.append(object_id)
        self.add_smothed_bbox()
        

    def remove_last_n_nodes(self, n):
        assert len(self.bboxes) == len(self.smothed_bboxes)
        for _ in range(n):
            self.bboxes.pop() 
            self.object_id.pop()
            self.smothed_bboxes.pop()

    def get_trajectory_length(self):
        # assert len(self.bboxes) == len(self.smothed_bboxes)
        return len(self.bboxes)


    def is_on_edge(self):
        '''
            is the last bbox on the image edge
        '''
        return is_bbox_on_edge(self.smothed_bboxes[-1], self.image_boundary)

    def get_leatest_node_center(self, is_smothed=True):
        assert self.get_trajectory_length() > 0
        if is_smothed:
            bbox = self.smothed_bboxes[-1]
        else:
            bbox = self.bboxes[-1]
        return int((bbox[0]+bbox[2])//2), int((bbox[1]+bbox[3])//2)

    def get_angle(self, is_smothed=True):
        if is_smothed:
            if len(self.smothed_bboxes) < 2:
                return 0
            bbox_c = self.smothed_bboxes[-1]
            bbox_l = self.smothed_bboxes[-2]
            x_c, y_c = (bbox_c[0] + bbox_c[2]) / 2, (bbox_c[1] + bbox_c[3]) / 2
            x_l, y_l = (bbox_l[0] + bbox_l[2]) / 2, (bbox_l[1] + bbox_l[3]) / 2
            return int(np.arctan2(y_c-y_l, x_c-x_l) * 180 / np.pi)
        else:
            if len(self.bboxes) < 2:
                return 0
            bbox_c = self.bboxes[-1]
            bbox_l = self.bboxes[-2]
            x_c, y_c = (bbox_c[0] + bbox_c[2]) / 2, (bbox_c[1] + bbox_c[3]) / 2
            x_l, y_l = (bbox_l[0] + bbox_l[2]) / 2, (bbox_l[1] + bbox_l[3]) / 2
            return int(np.arctan2(y_c-y_l, x_c-x_l) * 180 / np.pi)


class Tracker():
    def __init__(self, image_boundary, traj_smoth_alpha=0.2, 
                lost_times_thresh=100, lost_times_thresh_edge=5, appear_times_thresh=5, assoc_score_thresh = 0.2, cost_metric='iou'):
        self.trajectories = []
        self.alive_index = []
        self.real_alive_index = []
        self.whole_real_alive_index = []
        self.dead_index = []
        self.traj_smoth_alpha = traj_smoth_alpha
        self.lost_times_thresh = lost_times_thresh
        self.lost_times_thresh_edge = lost_times_thresh_edge
        self.appear_time_thresh = appear_times_thresh
        self.image_boundary = image_boundary
        self.assoc_score_thresh = assoc_score_thresh
        self.cost_metric = cost_metric


    def add_trajectory(self, start_bbox, start_object_id, born_time):
        ''', 
            Params:
                start_bboxes: shape [n,4]
                born_times: shape [n,], start frame number
        '''

        self.alive_index.append(len(self.trajectories)) # newly added trajectories are alive
        self.trajectories.append(Trajectory(start_bbox, start_object_id, born_time, self.image_boundary, self.traj_smoth_alpha))


    def association(self, candidate_bboxes):
        last_boxes = self.get_alive_last_bboxes(is_smoth=False)
        print('candidate trajectories:', last_boxes.shape[0])
        print('candidate objects:', candidate_bboxes.shape[0])
        association_costs = compute_cost_matrix(last_boxes, candidate_bboxes, metric=self.cost_metric)
        print('association_costs')
        print(association_costs)
        receivers_ids, providers_ids = linear_sum_assignment(association_costs) # ids in self.alive_index
        # receivers_ids, providers_ids = self.post_association_filter(association_costs, receivers_ids, providers_ids)

        return receivers_ids, providers_ids
    
    def post_association_filter(self, association_costs, receivers_ids, providers_ids):
        '''
            flitering unreasonable match after association
        '''
        filtered_receivers_ids = []
        filtered_providers_ids = []
        for receivers_id, providers_id in zip(receivers_ids, providers_ids):
            # only associations whose score are greater than assoc_score_thresh are valid
            if self.cost_metric == 'iou':
                if -association_costs[receivers_id][providers_id] > self.assoc_score_thresh:
                    filtered_receivers_ids.append(receivers_id)
                    filtered_providers_ids.append(providers_id)
            else:
                if association_costs[receivers_id][providers_id] < self.assoc_score_thresh:
                    filtered_receivers_ids.append(receivers_id)
                    filtered_providers_ids.append(providers_id)
        return np.array(filtered_receivers_ids), np.array(filtered_providers_ids)

    def update_tracker(self, candidate_bboxes_original, time_stamp):
        '''
            params:
                candidate_bboxes: numpy array with shape [n,4]
        '''
        # l=[]
        # for i, bbox in enumerate(candidate_bboxes_original):
        #     if not is_bbox_out_edge(bbox, self.image_boundary):
        #         l.append(i)
        candidate_bboxes = candidate_bboxes_original
        if len(self.alive_index) == 0: # no alive trajectories right now
            for object_id, bbox in enumerate(candidate_bboxes):
                # if is_bbox_on_edge(bbox, self.image_boundary) or time_stamp == 0:
                self.add_trajectory(start_bbox=bbox, start_object_id=object_id, born_time=time_stamp)
                print('+++ add new trajectory...')
            return 

        receivers_ids, providers_ids = self.association(candidate_bboxes)#?
        self.add_nodes_to_trajs(receivers_ids, providers_ids, candidate_bboxes)

        candidates_number = candidate_bboxes.shape[0]
        # receviders_number = receivers_ids.shape[0]
        # print('candidates_number:', candidates_number, 'receviders_number:', receviders_number)
        # print('receivers_ids', receivers_ids)

        # get lost_ids, indexes of trajectories not associated this time
        lost_ids = []                       # ids in self.alive_index
        lost_ids2 = []
        for i in range(len(self.alive_index)):
            if i not in receivers_ids:
                lost_ids.append(i)
      
        # processing trajectories not associated
        
        for i in range(len(lost_ids)):
            lost_ids2.append(lost_ids[-(i+1)])
            
        
        for lost_id in lost_ids2:
            
            # print('lost id:', self.alive_index[lost_id])
            traj = self.trajectories[self.alive_index[lost_id]]
            traj.lost_times += 1
            # print('lost time::', traj.lost_times, self.trajectories[self.alive_index[lost_id]].lost_times)
            traj.add_node(traj.bboxes[-1], -1) # add last bbox or predicted bbox

            # trajectory with last bbox on the edge and lost_times excceed times thresh will be dead
            if traj.lost_times > self.lost_times_thresh or\
                    (traj.lost_times > 1 and traj.get_trajectory_length() < 10):
                traj.is_alive = False
                self.alive_index.pop(lost_id)#?it will cause list index out of range in line132 around,solve it by reverse the list lost_ids
                traj.remove_last_n_nodes(traj.lost_times)

        # process unassociated candidate bboxes
        # get unassociated bboxes ids
        # unassociated_bbox_ids = []
        for i in range(candidates_number):
            if i not in providers_ids:

                # if is_bbox_on_edge(candidate_bboxes[i], self.image_boundary):
                self.add_trajectory(candidate_bboxes[i], i, time_stamp)
                print('unassociated candidate bboxes:', i)
                print('+++ add new trajectory...')
                # unassociated_bbox_ids.append(i)

                # if is_bbox_on_edge(candidate_bboxes[i,:], self.image_boundary):
                #     print('add new trajectory')
                #     self.add_trajectory(candidate_bboxes[i], time_stamp)

        self.real_alive_index = []
        for i in range(len(self.alive_index)):#appear test
            traj = self.trajectories[self.alive_index[i]]
            if traj.appear_time > self.appear_time_thresh:
                self.real_alive_index.append(self.alive_index[i])
                if self.whole_real_alive_index.count(self.alive_index[i]) == 0:
                    self.whole_real_alive_index.append(self.alive_index[i])
        print('alive_index:', self.alive_index)
        print("real:", self.real_alive_index)
        print('whole_real_alive_index:', self.whole_real_alive_index)

    def add_nodes_to_trajs(self, receivers_ids, providers_ids, candidate_bboxes):
        for re_id, pro_id in zip(receivers_ids, providers_ids):
            # print('='*100)
            # print(len(self.trajectories), len(self.alive_index), candidate_bboxes.shape[0])
            # print(re_id, pro_id)
            # print(self.alive_index[re_id])
            self.trajectories[self.alive_index[re_id]].add_node(candidate_bboxes[pro_id], pro_id)
            self.trajectories[self.alive_index[re_id]].lost_times = 0
            self.trajectories[self.alive_index[re_id]].appear_time += 1

    def get_alive_last_bboxes(self, is_smoth=True):
        '''
            get last one bboxees in all alive trajectories 
            return a numpy array with shape [len(alive_index), 4]
        '''
        last_bboxes = []
        for alive_idx in self.alive_index:
            print('alive_idx:', alive_idx)
            if is_smoth:
                last_bboxes.append(self.trajectories[alive_idx].smothed_bboxes[-1])
            else:
                last_bboxes.append(self.trajectories[alive_idx].bboxes[-1])
        last_bboxes = np.array(last_bboxes)
        # print(last_bboxes.shape)
        # last_bboxes = last_bboxes.reshape([last_bboxes.shape[1], last_bboxes.shape[2]])

        return last_bboxes

if __name__ == '__main__':
    tracker = Tracker()

