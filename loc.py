import cv2
import numpy as np
from scipy.cluster.vq import kmeans

class blobs:
    def __init__(self, min_area = 100):

        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True;
        params.minCircularity = 0.5;
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        params.filterByArea = True
        params.minArea = min_area

        params.minRepeatability = 1
        params.minThreshold = 200
        params.thresholdStep = 50
        self.detector = cv2.SimpleBlobDetector_create(params)
    def blob_location(self, im):
        if im.dtype != np.uint8:
            raise
        keypoints = self.detector.detect(im)
        coords = np.array(list(map(lambda x: x.pt, keypoints)))
        return coords


def rotate_blobs(blobs, angle, old_dims, new_dims):
    if blobs.shape[-1] != 2:
        raise
    if len(blobs.shape) != 2:
        raise

    old_middle = [old_dims[1] // 2, old_dims[0] // 2]
    imr_middle = [new_dims[1] // 2, new_dims[0] // 2]

    blob_vec_middle = blobs - old_middle

    rotation_matrix = cv2.getRotationMatrix2D((0,0), angle, 1.0)

    new_blob_vec_middle = np.dot(blob_vec_middle, rotation_matrix)[:, 0:2]

    new_blobs = new_blob_vec_middle + imr_middle

    return new_blobs


def sort_points(blobs, column_number = 4):
    ''' organize blob points according to patern'''
    if blobs.shape[-1] != 2:
        raise

    clusters = np.sort(kmeans(blobs[:,0], column_number)[0])

    columns = [ [] for _ in range(column_number) ]


    for blob in blobs:
        column_i = np.argmin(np.abs(clusters - blob[0]))
        columns[column_i].append(blob)

    for i in range(len(columns)):
        columns[i] = np.array(columns[i])
        columns[i] = columns[i][columns[i][:,1].argsort()]

        #print(i, columns[i].shape)

    return np.array(columns)



def locate_4_outer_points(arr):
    '''using Pythagoras theorem'''
    if len(arr.shape) != 2:
        raise
    mean_point = np.mean(arr,axis=0)
    difference = arr - mean_point
    distance_from_mean = np.sqrt(np.sum(np.square( difference) , axis=1))
    indexes = np.argpartition(distance_from_mean, -4)[-4:]
    return arr[indexes]
def sides_from_outer_points(arr):

    if arr.shape != (4,2):
        raise

    sorted_points = arr[arr[:,0].argsort()]
    left_points = sorted_points[:2]
    right_points = sorted_points[2:]
    #print(sorted_points)
    #print("points",left_points)
    #print("right", right_points)

    left_base = left_points[np.argmin(np.sum(left_points, axis=1))]
    left_other = left_points[np.argmax(np.sum(left_points,axis=1))]
    right_base = right_points[np.argmin(np.sum(right_points, axis=1))]
    right_other = right_points[np.argmax(np.sum(right_points,axis=1))]

    return {"left":(left_base, left_other),"right":(right_base,right_other)}







def angle_2_points(main_point, second_point):
    if main_point.shape != second_point.shape:
        raise
    if main_point.shape != (2,):
        raise
    sides = main_point - second_point
    degrees = np.degrees(np.arctan(sides[0] / sides[1]))
    return degrees

