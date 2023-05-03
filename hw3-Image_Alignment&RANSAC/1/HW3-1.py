import cv2
import copy
import random
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return "({0}, {1})".format(self.y, self.x)

def ComputeHomography(points1, points2):
    '''
    Solving the Homographies
    find the solution of Ah = 0 via least square method
    to solve h w.r.t Ah = 0 is to minimize |Ah - 0|^2
    Homography h is equal to the eigenvector of A.T * A with the smallest eigenvalue

    where A.shape=(2n,9) h.shape=(9,1) n=len(point1)
    '''

    n = len(points1)

    A = np.zeros((2*n,9))
    h = np.zeros((9,1))
    H = np.zeros((3,3))

    # Construct the matrix A according to the Homoraphy Formula
    for i in range(n):
        pt1 = points1[i]
        pt2 = points2[i]
        A[2*i,:] = np.array([pt1.x, pt1.y, 1, 0, 0, 0, -(pt2.x)*(pt1.x), -(pt2.x)*(pt1.y), -(pt2.x)])
        A[2*i+1,:] = np.array([0, 0, 0, pt1.x, pt1.y, 1, -(pt2.y)*(pt1.x), -(pt2.y)*(pt1.y), -(pt2.y)])

    # find eginvector w.r.t min EigenValue
    ATA = np.array(np.matmul(A.T, A))
    egi_val, egi_vector = np.linalg.eig(ATA)
    min_egival_idx = np.where(egi_val == np.amin(egi_val)) # find minium EgienValue of A.T * A
    h = egi_vector[:, min_egival_idx].flatten()
    
    # find H (Homography Transform Matrix)
    H[0] = np.array([h[0], h[1], h[2]])
    H[1] = np.array([h[3], h[4], h[5]])
    H[2] = np.array([h[6], h[7], h[8]])

    print('Homography Transform Matrix:')
    print(H)

    return H

def draw_selectedArea(img, points):
    img = copy.deepcopy(img)
    
    start_point, end_point = (points[0].x, points[0].y), (points[1].x, points[1].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2)

    start_point, end_point = (points[1].x, points[1].y), (points[2].x, points[2].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2) 

    start_point, end_point = (points[2].x, points[2].y), (points[3].x, points[3].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2) 

    start_point, end_point = (points[3].x, points[3].y), (points[0].x, points[0].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=4) 
    cv2.imwrite('./2/output/selected_img.jpg', img)



def SIFTFeaturMatching(img1, img2, out_path, threshold=0.58):
    
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2,None)
    
    matches = []
    src_pts = []
    dst_pts = []

    # brute force find euclidian distance between two keypoints(discriptor)
    for idx1, val1 in enumerate(descriptors1):

        distance = []

        for idx2 in range(0, descriptors2.shape[0]):
            dis = np.linalg.norm(descriptors2[idx2] - val1, ord=1)
            distance.append([dis, idx2])
        
        distance = sorted(distance, key = lambda x: x[0])

        if  distance[0][0] < threshold * distance[1][0]:
            idx2 = distance[0][1]
            dis = distance[0][0]
            matches.append(cv2.DMatch(idx1, idx2, dis))
            src_pts.append(Point(keypoints1[idx1].pt[0], keypoints1[idx1].pt[1]))
            dst_pts.append(Point(keypoints2[idx2].pt[0], keypoints2[idx2].pt[1]))
            
    #matches = sorted(matches, key = lambda x:x.distance)

    print(len(matches))

    result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(out_path, result)

    return np.array(src_pts), np.array(dst_pts), matches, keypoints1, keypoints2

def RANSAC(src_pts, dst_pts, matches, threshold=0.12):
   
    outlier = 0
    inlier_matches = []

    while True:
        n = len(src_pts)
        
        idx = [i for i in range(n)]
        random.shuffle(idx)
        idx = idx[0:4]
        
        src_ = [pt for pt in src_pts[idx]]
        dst_ = [pt for pt in dst_pts[idx]]
        
        H = ComputeHomography(src_, dst_)

        outlier = 0

        for i in range(n):
            src_pt = np.array([src_pts[i].x, src_pts[i].y, 1])

            x, y, z = np.matmul(H, src_pt)

            dst_pt = np.array([dst_pts[i].x, dst_pts[i].y])
            predict_pt = np.array([x/z, y/z])

            dis = np.linalg.norm(dst_pt-predict_pt)

            if dis > 1.5:
                outlier += 1
            else:
                inlier_matches.append(matches[i])

        if outlier / n  < threshold:
            M = H
            break
        else:
            inlier_matches.clear()

    return M, inlier_matches

def draw_result(img1, img2, inlier_matches, keypoints1, keypoints2, H, corner, output_path):
    
    h,w = img1.shape

    dsts = np.zeros((4, 1, 2))
    for i in range(len(corner)):
        x, y = corner[i]
        x_, y_, z_ = np.matmul(H, np.array([x, y, 1]).T)
        dsts[i] = np.array([[x_/z_, y_/z_]])

    out_img = cv2.polylines(img2,[np.int32(dsts)],True,255,3, cv2.LINE_AA)
    result = cv2.drawMatches(img1, keypoints1, out_img, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(output_path, result)

def solve():

    corner = [[[16, 44], [12, 315], [435, 314], [420, 42]], 
                [[15, 31], [23, 320], [424, 314], [407, 14]],
                [[23, 30], [13, 300], [423, 295], [413, 28]]]

    for i in range(1, 3+1):
        img1 = cv2.imread("./1/1-book{0}.jpg".format(i),cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread("./1/1-image.jpg",cv2.IMREAD_GRAYSCALE)
        
        src_pts, dst_pts, matches, keypoints1, keypoints2 = SIFTFeaturMatching(img1, img2, out_path='./1/output/SIFT-book{0}.jpg'.format(i), threshold=0.58)
        
        H, inlier_matches = RANSAC(src_pts, dst_pts, matches, threshold=0.12)
        draw_result(img1, img2, inlier_matches, keypoints1, keypoints2, H=H, corner=corner[i-1], output_path='./1/output/RANSAC-book{0}.jpg'.format(i))


solve()