import cv2
import numpy as np


def sift(RGB, gray):

    sift = cv2.SIFT_create(contrastThreshold = 0.2)
    kp = sift.detect(gray,None)
    print(len(kp))

    for kp in kp:
        x, y = kp.pt
        x = int(round(x))
        y = int(round(y))

        # Draw a cross with (x, y) center
        img = cv2.drawMarker(RGB, (x, y), color=(0,0,255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)

    cv2.imwrite('./2/output/a_SIFT_InterestPoint.jpg',img)
    #return img

def SIFTFeaturMatching(img1, img2):
    
    sift = cv2.SIFT_create(contrastThreshold = 0.16)

    keypoints1, descriptors1 = sift.detectAndCompute(img1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2,None)
    
    matches = []

    # brute force find euclidian distance between two keypoints(discriptor)
    for idx1, val1 in enumerate(descriptors1):

        distance = []

        for idx2 in range(0, descriptors2.shape[0]):
            dis = np.linalg.norm(descriptors2[idx2] - val1, ord=1)
            distance.append([dis, idx2])

        distance = sorted(distance, key = lambda x: x[0])
        
        if  distance[0][0] > 0.2 * distance[1][0]:
            matches.append(cv2.DMatch(idx1, distance[0][1], distance[0][0]))
        
        del distance
    
    matches = sorted(matches, key = lambda x:x.distance)

    result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None)
    
    cv2.imwrite('./2/output/b_SIFT_Mathcing.jpg', result)

def solve():
    BGRimage = cv2.imread("./2/1a_notredame.jpg")
    Grayimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2GRAY)
    BGRimage_2 = cv2.imread("./2/1b_notredame.jpg")
    Grayimage_2 = cv2.cvtColor(BGRimage_2, cv2.COLOR_BGR2GRAY)
    sift(BGRimage, Grayimage)
    SIFTFeaturMatching(Grayimage, Grayimage_2)

solve()