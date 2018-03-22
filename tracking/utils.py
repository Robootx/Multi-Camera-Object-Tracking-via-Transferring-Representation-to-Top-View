import numpy as np

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


def compute_cost_matrix(bbox_receiver, bbox_provider, metric='distance'):
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
        # receiver_center = (bbox_receiver[:,0:2] + bbox_receiver[:, 2:4]) / 2
        # provider_center = (bbox_provider[:,0:2] + bbox_provider[:, 2:4]) / 2
        #print(bbox_receiver.shape, bbox_provider.shape)
        #print('-'*60)
        extend_bbox_receiver = np.repeat(bbox_receiver, provider_length, axis = 0)
        extend_bbox_provider = np.tile(bbox_provider, (receiver_length,1))
        #print(extend_bbox_receiver.shape, extend_bbox_provider.shape)
        cost_matrix = distance(extend_bbox_receiver, extend_bbox_provider)
        cost_matrix = cost_matrix.reshape([receiver_length,provider_length])
    else:
        raise
    return cost_matrix

def is_bbox_on_edge(bbox, edges):
    '''
        edges: a bbox represent valid area [x1, y1, x2, y2]
    '''
    # print(edges)
    # print(bbox)
    return False
    # return (bbox[0] <= edges[0] or bbox[1] <= edges[1] \
    #         or bbox[2] >= edges[2] or bbox[3] >= edges[3])

def is_bbox_out_edge(bbox, edges):
    '''
        edges: a bbox represent valid area [x1, y1, x2, y2]
    '''
    # print(edges)
    # print(bbox)
    return False
    # return (bbox[2] <= edges[0] or bbox[3] <= edges[1] \
    #         or bbox[0] >= edges[2] or bbox[1] >= edges[3])
