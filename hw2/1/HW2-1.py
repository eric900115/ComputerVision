import numpy as np
import math
import cv2
import copy
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def InputPoints():

    f = open('./1/pt_2D_1.txt', 'r')

    points1 = []
    for val in f.readlines()[1:]:
        x, y = val.split()
        points1.append(Point(float(x), float(y)))

    f.close()

    f = open('./1/pt_2D_2.txt', 'r')

    points2 = []
    for val in f.readlines()[1:]:
        x, y = val.split()
        points2.append(Point(float(x), float(y)))

    f.close()

    return points1, points2

def EightPointAlgo(points1, points2):

    n = len(points1)

    W = np.zeros((n, 9))
    f = np.zeros((9,1))

    # construct Matrix W
    for i in range(n):
        u, v = points1[i].x, points1[i].y
        u_, v_ = points2[i].x, points2[i].y
        W[i] = np.array([u*u_, u*v_, u, v*u_, v*v_, v, u_, v_, 1])

    # calculate Approximation F (F_) By least square method 
    # the least square solution is equal to  
    F_ = np.zeros((3, 3))
    U0, s0, v_T = np.linalg.svd(W, full_matrices=True)
    F_ = np.array([np.array([v_T[-1][0], v_T[-1][1], v_T[-1][2]]),
                    np.array([v_T[-1][3], v_T[-1][4], v_T[-1][5]]),
                    np.array([v_T[-1][6], v_T[-1][7], v_T[-1][8]])])
    
    '''
    claculate Fundemantal Matrix F
    To calculte ||F - F_||=0 Subject to det(F)=0 (rank(F)=2)
    Solved By SVD, SVD(F_)=U*S*V_T
    F = U*S_*V_T
    S_ : 1.S_[i][j]==S[i][j] when i!=2 and j!=2 2.S[2][2]=0(Since Rank(F)=2)
    '''

    S1 = np.zeros((3,3))
    U1, s1, V1_T = np.linalg.svd(F_, full_matrices=True)
    S1[0][0] = s1[0]
    S1[1][1] = s1[1]
    F = np.matmul(np.matmul(U1, S1), V1_T)

    return F

def normalizedPoint(point):

    normalized_point = []

    n = len(point)

    centroid_x = np.sum([pt.x for pt in point])/len(point)
    centroid_y = np.sum([pt.y for pt in point])/len(point)

    distance_sum = 0

    for pt in point:
        distance_sum += math.sqrt((pt.x - centroid_x)**2 + (pt.y - centroid_y)**2)

    mean_distance = distance_sum/len(point)

    scale = math.sqrt(2)/mean_distance

    A = np.array([np.array([scale, 0, -scale*centroid_x]),
                np.array([0, scale, -scale*centroid_y]),
                np.array([0, 0, 1])])


    for i in range(n):
        pt = np.array([point[i].x, point[i].y, 1])
        new_pt = np.matmul(A, pt)
        normalized_point.append(Point(new_pt[0]/new_pt[2], new_pt[1]/new_pt[2]))

    return normalized_point, A

def normalized_eight_point(points1, points2):
    
    normalized_pts1, A1 = normalizedPoint(points1)
    normalized_pts2, A2 = normalizedPoint(points2)
    F_q = EightPointAlgo(normalized_pts1, normalized_pts2)
    F = np.matmul(np.matmul(A1.T, F_q), A2)

    return F


def Plot_EpipolarLine(points1, points2, img1, img2, F, is_normalized):

    #line = np.matmul(F, points2.T).T
    n = len(points1)

    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)
    
    for i in range(n):
        
        # line is the direction vector of epipolor line that pass through pt
        pt = points1[i]
        pt2 = points2[i]
        line = np.matmul(F, np.array([pt2.x, pt2.y, 1])).T

        # ax + by + c = 0
        # where a = line[0], b = line[1], c = line[2]
        a, b, c = line[0], line[1], line[2]

        # a*0 + by + c = 0 => y = -c/b
        x_start, y_start = 0, int(-c/b)
        start_point = (x_start, y_start)

        # a*img.shape[1] + by + c = 0 => y = -(c + a*img.shape[1])/b
        x_end, y_end = img1.shape[1], int(-(a*img1.shape[1]+c)/b)
        end_point = (x_end, y_end)

        cv2.line(img1, start_point, end_point, color=(255, 0, 0), thickness=2) 
        
        img1 = cv2.circle(img1, (int(pt.x), int(pt.y)), radius=5, color=(0, 0, 255), thickness=-1)
    
    if is_normalized == True:
        cv2.imwrite('./1/output/b_img1.jpg', img1)
    else:
        cv2.imwrite('./1/output/a_img1.jpg', img1)

    #line2 = np.matmul(F.T, points1.T).T
    for i in range(n):

        # line is the direction vector of epipolor line that pass through pt
        pt = points1[i]
        pt2 = points2[i]
        line = np.matmul(F.T, np.array([pt.x, pt.y, 1]).T).T

        # ax + by + c = 0
        # where a = line[0], b = line[1], c = line[2]
        a, b, c = line[0], line[1], line[2]

        # a*0 + by + c = 0 => y = -c/b
        x_start, y_start = 0, int(-c/b)
        start_point = (x_start, y_start)

        # a*img.shape[1] + by + c = 0 => y = -(c + a*img.shape[1])/b
        x_end, y_end = img2.shape[1], int(-(a*img2.shape[1]+c)/b)
        end_point = (x_end, y_end)

        cv2.line(img2, start_point, end_point, color=(255, 0, 0), thickness=2) 
        
        img1 = cv2.circle(img2, (int(pt2.x), int(pt2.y)), radius=5, color=(0, 0, 255), thickness=-1)
    
    if is_normalized == True:
        cv2.imwrite('./1/output/b_img2.jpg', img2)
    else:
        cv2.imwrite('./1/output/a_img2.jpg', img2)
    
def dist_Point_2_Epipolorline(points1, points2, F):

    n = len(points1)

    dist = 0

    for i in range(n):
        # line is the direction vector of epipolor line that pass through pt
        pt = points1[i]
        pt2 = points2[i]
        line = np.matmul(F, np.array([pt2.x, pt2.y, 1]).T).T

        # ax + by + c = 0
        # where a = line[0], b = line[1], c = line[2]
        a, b, c = line[0], line[1], line[2]

        # dist(i, line) = |a*x0 + b*y0 + c| / sqrt(a^2 + b^2)
        dist += abs(a*pt.x + b*pt.y + c) / math.sqrt(a**2 + b**2)

    avg_dist = dist / n

    return avg_dist

points1, points2 = InputPoints()

img1 = cv2.imread('1/image1.jpg')
img2 = cv2.imread('1/image2.jpg')

F = EightPointAlgo(points1, points2)
F_normalized = normalized_eight_point(points1, points2)

is_normalize = False
Plot_EpipolarLine(points1, points2, img1, img2, F, is_normalize)

is_normalize = True
Plot_EpipolarLine(points1, points2, img1, img2, F_normalized, is_normalize)

dist_unnormalize1 = dist_Point_2_Epipolorline(points1, points2, F)
dist_unnormalize2 = dist_Point_2_Epipolorline(points2, points1, F.T)

dist_normalized1 = dist_Point_2_Epipolorline(points1, points2, F_normalized)
dist_normalized2 = dist_Point_2_Epipolorline(points2, points1, F_normalized.T)

print('Fundemantal Matrix (unnormalized)')
print(F)
print('Fundemantal Matrix (normalized)')
print(F_normalized)

print('the Average distance of the points to epipolar line in image 1 (unnormalized) is ', dist_unnormalize1)
print('the Average distance of the points to epipolar line in image 1 (normalized) is ', dist_normalized1)

print('the Average distance of the points to epipolar line in image 2 (unnormalized) is ', dist_unnormalize2)
print('the Average distance of the points to epipolar line in image 2 (normalized) is ', dist_normalized2)