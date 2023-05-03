from math import ceil, floor
import numpy as np
import cv2
import copy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

def Backward_and_Bilinear_Wrapping(img, H):
    '''
    Implement Image wrapping through Backward Wrapping and Bilinear Interpolation
    
    input: 
        img : image to be wrapping
        H(3*3) : Wrapping Matrix
    
    output:
        new_img : image after wrapping
    '''

    new_img = np.zeros((1800, 1800, 3))

    H_Inverse = np.linalg.inv(H)

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            for k in range(new_img.shape[2]):
                '''
                Forward wrapping : use the point in origin image to find the correspond point in ouput image
                [new_x, new_y, new_z] = H * [x, y, 1]

                Backward wrapping : use the point in ouput image to find the correspond point in origin image
                [x, y, 1] = H_inverse * [new_x, new_y, new_z]

                note : [new_x, new_y, new_z] is in Homogeneous Coordinate
                '''

                # Implementation of BackWard Wrapping
                new_y, new_x = i, j
                x, y, z = np.matmul(H_Inverse, np.array([new_x, new_y, 1]))

                # convert Homogeneous Cordinate to Euclidean Coordinate
                x = x / z
                y = y / z
                
                round_x = int(round(x))
                round_y = int(round(y))

                ceil_x = int(ceil(x))
                ceil_y = int(ceil(y))

                # Bilinear Interpolation
                if round_x-1 >= 0 and round_x+1 < img.shape[1] and round_y-1 >= 0 and round_y+1 < img.shape[0]:

                    Q11 = img[round_y, round_x, k] # pixel value of Point(round_x, round_y)
                    Q21 = img[round_y, ceil_x, k]  # pixel value of Point(ceil_x, round_y)
                    Q12 = img[ceil_y, round_x, k]  # pixel value of Point(round_x, ceil_y)
                    Q22 = img[ceil_y, ceil_x, k]   # pixel value of Point(ceil_x, ceil_y)

                    if ceil_x-round_x > 0:
                        R1 = Q11*(ceil_x-x)/(ceil_x-round_x) + Q21*(x-round_x)/(ceil_x-round_x)  # pixel value of Point(X, round_y)
                        R2 = Q12*(ceil_x-x)/(ceil_x-round_x) + Q22*(x-round_x)/(ceil_x-round_x)  # pixel value of Point(X, ceil_y)
                    else:
                        R1 = Q11 # When ceil_x == round_x, R1 = Q11 = Q21
                        R2 = Q12 # When ceil_x == round_x, R2 = Q12 = Q22
                    
                    if ceil_y-round_y > 0:
                        P = R1*(ceil_y-y)/(ceil_y-round_y) + R2*(y-round_y)/(ceil_y-round_y)     #pixel value of Point(X, y)
                    else:
                        P = R1 # When ceil_y == round_y, P = R1 = R2 (Since Q11 == Q12 and Q21 == Q22)
                    new_img[new_y, new_x, k] = P

    cv2.imwrite('./2/output/rectified_img.jpg', new_img[350:1350, 350:1350])

    return new_img

def draw_selectedArea(img, points):
    img = copy.deepcopy(img)
    
    start_point, end_point = (points[0].x, points[0].y), (points[1].x, points[1].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2)

    start_point, end_point = (points[1].x, points[1].y), (points[2].x, points[2].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2) 

    start_point, end_point = (points[2].x, points[2].y), (points[3].x, points[3].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2) 

    start_point, end_point = (points[3].x, points[3].y), (points[0].x, points[0].y)
    cv2.line(img, start_point, end_point, color=(255, 0, 0), thickness=2) 
    cv2.imwrite('./2/output/selected_img.jpg', img)

src_points = [Point(418, 800), Point(896, 1016), Point(886, 65), Point(435, 342)]
taget_points = [Point(500, 1200), Point(1200, 1200), Point(1200, 500), Point(500, 500)]

img = cv2.imread('2/Delta-Building.jpg')
draw_selectedArea(img, src_points)

H = ComputeHomography(src_points, taget_points)
Backward_and_Bilinear_Wrapping(img, H)