import random
import math
import cv2
import copy
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import numpy as np

def get_centroid(img, height, width, k):

    centroid = np.zeros((k, 3))

    for i in range(k):
        y = random.randrange(height)
        x = random.randrange(width)
        centroid[i] = np.array([img[y][x][0], img[y][x][1], img[y][x][2]])

    return centroid

def plot_segment():
    pass

def kmeans(img, k, out_path):

    img = copy.deepcopy(img)

    h = img.shape[0]
    w = img.shape[1]

    min_loss = 10000
    best_center = []

    for s in range(50):

        centroid = get_centroid(img, h, w, k)
        centroid = centroid.astype(np.float64)

        while True: 
            # loop until the centroid of all clusters are not changing

            pt_cluster = np.zeros((h, w))
            
            # find the cluster that each point belong to
            for i in range (h):
                for j in range (w):
                    val = img[i][j]
                    dis = [np.linalg.norm(val-centroid[c]) for c in range(k)]
                    pt_cluster[i][j] = np.argmin(dis)
            pt_cluster = pt_cluster.astype(np.int8)
                    
            # update center
            new_center = np.zeros((k, 3)).astype(np.float64)
            sum_dis = np.zeros((k, 3)).astype(np.float64)
            num_pt_cluster = np.zeros((k))
            
            for i in range (h):
                for j in range (w):
                    cluster = pt_cluster[i][j]
                    num_pt_cluster[cluster] += 1
                    sum_dis[cluster][0] += img[i][j][0]
                    sum_dis[cluster][1] += img[i][j][1]
                    sum_dis[cluster][2] += img[i][j][2]
            
            #print(num_pt_cluster)

            for i in range(k):
                new_center[i][0] = sum_dis[i][0] / num_pt_cluster[i]
                new_center[i][1] = sum_dis[i][1] / num_pt_cluster[i]
                new_center[i][2] = sum_dis[i][2] / num_pt_cluster[i]
                
                    
            # check covergence
            #if np.array_equal(new_center.astype(np.int32), centroid.astype(np.int32)):
            if np.sum(np.linalg.norm(new_center.astype(np.int32) - centroid.astype(np.int32), ord=1)) < 2*k:
                centroid = new_center
                #print('------')
                break
            else:
                #print(centroid.astype(np.int32))
                centroid = new_center
                
                #print(new_center.astype(np.int32))
                #print('------')
        
        # calculate loss
        loss = 0
        for i in range(h):
            for j in range(w):
                val = img[i][j]
                dis = [np.linalg.norm(val-centroid[c]) for c in range(k)]
                loss += np.amin(dis)
                
        print(loss, s)
        if s == 0 or loss < min_loss:
            min_loss = loss
            best_center = copy.deepcopy(centroid)
    
    # plot the segmentation image
    
    for i in range(h):
        for j in range(w):
            val = img[i][j]
            dis = [np.linalg.norm(val-best_center[c]) for c in range(k)]
            cluster = np.argmin(dis)
            #print(cluster)
            img[i][j] = best_center[cluster]
    cv2.imwrite(out_path, img)
    '''
    for s in range(1):

        centroid = get_centroid(img, h, w, k)
        centroid = centroid.astype(np.float64)

        while True: 
            # loop until the centroid of all clusters are not changing

            pt_cluster = np.zeros((h, w))
            
            # find the cluster that each point belong to
            for i in range (h):
                for j in range (w):
                    val = img[i][j]
                    dis = [np.linalg.norm(val-centroid[c]) for c in range(k)]
                    pt_cluster[i][j] = np.argmin(dis)
            pt_cluster = pt_cluster.astype(np.int8)
                    
            # update center
            new_center = np.zeros((k, 3)).astype(np.float64)
            sum_dis = np.zeros((k, 3)).astype(np.float64)
            num_pt_cluster = np.zeros((k))
            
            for i in range (h):
                for j in range (w):
                    cluster = pt_cluster[i][j]
                    num_pt_cluster[cluster] += 1
                    sum_dis[cluster][0] += img[i][j][0]
                    sum_dis[cluster][1] += img[i][j][1]
                    sum_dis[cluster][2] += img[i][j][2]
            
            for i in range(k):
                new_center[i][0] = sum_dis[i][0] / num_pt_cluster[i]
                new_center[i][1] = sum_dis[i][1] / num_pt_cluster[i]
                new_center[i][2] = sum_dis[i][2] / num_pt_cluster[i]
                
                    
            # check covergence
            if np.array_equal(new_center.astype(np.int32), centroid.astype(np.int32)):
            #if np.sum(np.linalg.norm(new_center.astype(np.int32) - centroid.astype(np.int32), ord=1)) < k:
                centroid = new_center
                print('------')
                break
            else:
                #print(centroid.astype(np.int32))
                centroid = new_center
                
                #print(new_center.astype(np.int32))
                #print('------')
        
        # calculate loss
        loss = 0
        for i in range(h):
            for j in range(w):
                val = img[i][j]
                dis = [np.linalg.norm(val-centroid[c]) for c in range(k)]
                loss += np.amin(dis)
                
        print(loss)
    '''
def kmeansPlusPlus(img, k, out_path):
    
    img = copy.deepcopy(img)

    h = img.shape[0]
    w = img.shape[1]

    centroid = np.zeros((k, 3))
    
    centroid[0] = get_centroid(img, h, w, 1)
    
    for c in range(1, k):
        
        distance_pt_to_NearstCentroid = np.zeros((h, w))
        
        # find
        for i in range(h):
            for j in range(w):
                val = img[i][j]
                dis = [np.linalg.norm(val-centroid[s]) for s in range(c)]
                min_dis = np.amin(dis)
                #print(dis)
                #print(min_dis)
                distance_pt_to_NearstCentroid[i][j] = min_dis
        
        # 
        next_centroid = np.unravel_index(distance_pt_to_NearstCentroid.argmax(), distance_pt_to_NearstCentroid.shape)
        print(next_centroid)
        c_y, c_x = next_centroid
        centroid[c] = img[c_y][c_x]
        
    # plot the segmentation image
    #color = [list(np.random.choice(range(256), size=3)) for i in range(k)]
    
    for i in range(h):
        for j in range(w):
            val = img[i][j]
            dis = [np.linalg.norm(val-centroid[c]) for c in range(k)]
            cluster = np.argmin(dis)
            #print(cluster)
            img[i][j] = centroid[cluster]
    cv2.imwrite(out_path, img)

    
def meanshift(img, bandwith, spatial=False, spatial_bandwith=35):
    
    h = img.shape[0]
    w = img.shape[1]

    centroid = []
    cluster = np.zeros((h, w))
    is_visited = np.zeros((h, w))

    print(h, w)

    for i in range(h):
        for j in range(w):

            if is_visited[i][j] == 1:
                continue
            
            if spatial == False:
                mean = np.zeros((3,))
                mean_prev = img[i][j]
            else:
                mean = np.zeros((5,))
                
                mean_prev = np.array([*img[i][j], *[i, j]])
            
            iteration = 0

            while True:

                # calculate new mean
                sum_color = np.zeros((3,))
                
                if spatial == True:
                    sum_spatial = np.zeros((2,))
                
                num_pts = 0

                for s in range(h):
                    for t in range(w):

                        if spatial == False:
                            color_dis = np.linalg.norm(img[s][t] - mean_prev)

                            if color_dis < bandwith:
                                is_visited[s][t] = 1
                                sum_color[0] += img[s][t][0]
                                sum_color[1] += img[s][t][1]
                                sum_color[2] += img[s][t][2]
                                num_pts += 1
                        else:
                            color_dis = np.linalg.norm(img[s][t] - mean_prev[0:3])
                            spatial_val = np.array([s, t])
                            spatial_dis = np.linalg.norm(spatial_val - mean_prev[3:5])

                            if color_dis < bandwith and spatial_dis < spatial_bandwith:
                                is_visited[s][t] = 1
                                sum_color[0] += img[s][t][0]
                                sum_color[1] += img[s][t][1]
                                sum_color[2] += img[s][t][2]
                                sum_spatial[0] += s
                                sum_spatial[1] += t
                                num_pts += 1

                
                if  spatial == False:
                    mean = sum_color / num_pts
                else:
                    mean[0:3] = sum_color / num_pts
                    mean[3:5] = sum_spatial / num_pts

                iteration += 1

                if spatial == False:
                    if np.linalg.norm(mean[0:3] - mean_prev[0:3]) < 0.05 * bandwith:
                        break
                    else:
                        mean_prev = mean
                else:
                    if np.linalg.norm(mean[0:3] - mean_prev[0:3]) < 0.05 * bandwith and np.linalg.norm(mean[3:5] - mean_prev[3:5]) < 0.05 * spatial_bandwith:
                        break
                    else:
                        mean_prev = mean
        
            color_dis_pt_centroid = np.array([np.linalg.norm(mean[0:3] - val[0:3]) for val in centroid])
            spatial_dis_pt_centroid = np.array([np.linalg.norm(mean[3:5] - val[3:5]) for val in centroid])


            if len(color_dis_pt_centroid) != 0 and np.amin(color_dis_pt_centroid) < bandwith and np.amin(spatial_dis_pt_centroid) < spatial_bandwith:
                pass
            else:
                # create a new cluster
                centroid.append(mean)

            print(i, j)
            print(centroid)

    print(centroid)
    for i in range(h):
        for j in range(w):
            
            if spatial == False:
                dis_pt_centroid = np.array([np.linalg.norm(val - img[i][j]) for val in centroid])
                centroid_id = np.argmin(dis_pt_centroid)
                print(centroid_id)
                img[i][j] = centroid[centroid_id]
            else:
                spatial_dis_pt_centroid = np.array([np.linalg.norm(val[3:5] - [i, j]) for val in centroid])
                color_dis_pt_centroid = np.array([np.linalg.norm(val[0:3] - img[i][j]) for val in centroid])

                min_spatial_dist = 10000000000
                centroid_id = -1

                for k in range(len(color_dis_pt_centroid)):
                    if color_dis_pt_centroid[k] < bandwith:
                        if spatial_dis_pt_centroid[k] < spatial_bandwith and spatial_dis_pt_centroid[k] < min_spatial_dist:
                            min_spatial_dist = spatial_dis_pt_centroid[k]
                            centroid_id = k
                
                print(centroid_id)
                img[i][j] = centroid[centroid_id][0:3]
    
    cv2.imwrite('./2/output12.jpg', img)

def MeanShift(img, bandwith, out_path):

    img = copy.deepcopy(img)

    h = img.shape[0]
    w = img.shape[1]

    centroid = []
    cluster = np.zeros((h, w))
    is_visited = np.zeros((h, w))

    print(h, w)

    for i in range(h):
        for j in range(w):

            if is_visited[i][j] == 1:
                continue

            mean = np.zeros((3,))
            mean_prev = np.array(img[i][j])
            
            iteration = 0

            while True:

                # calculate new mean
                sum_dis = np.zeros((3,))
                
                num_pts = 0

                for s in range(h):
                    for t in range(w):
                        dis = np.linalg.norm(img[s][t] - mean_prev)
                        #print(dis)
                        if dis < bandwith:
                            is_visited[s][t] = 1
                            sum_dis[0] += img[s][t][0]
                            sum_dis[1] += img[s][t][1]
                            sum_dis[2] += img[s][t][2]
                            num_pts += 1
                #print(num_pts)
                mean = sum_dis / num_pts

                iteration += 1

                if np.linalg.norm(mean - mean_prev) < 0.05 * bandwith:
                    break
                else:
                    mean_prev = mean
        
            dis_pt_centroid = np.array([np.linalg.norm(mean - val) for val in centroid])


            if len(dis_pt_centroid) != 0 and np.amin(dis_pt_centroid) < bandwith:
                pass
            else:
                # create a new cluster
                centroid.append(mean)

            print(i, j)
            #print(centroid)

    print(centroid)
    for i in range(h):
        for j in range(w):
            dis_pt_centroid = np.array([np.linalg.norm(val - img[i][j]) for val in centroid])
            centroid_id = np.argmin(dis_pt_centroid)
            #print(centroid_id)
            img[i][j] = centroid[centroid_id]
           
    cv2.imwrite(out_path, img)

def MeanShift_Spatial_and_Color(img, bandwith, out_path):

    img = copy.deepcopy(img)
    
    h = img.shape[0]
    w = img.shape[1]

    centroid = []
    cluster = np.zeros((h, w))
    is_visited = np.zeros((h, w))

    print(h, w)

    for i in range(h):
        for j in range(w):

            if is_visited[i][j] == 1:
                continue

            mean = np.zeros((5,))
            mean_prev = np.array([*(img[i][j]/255), *[i/h, j/w]])
            
            iteration = 0

            while True:

                # calculate new mean
                sum_dis = np.zeros((5,))
                
                num_pts = 0

                for s in range(h):
                    for t in range(w):
                        dis = np.linalg.norm([*(img[s][t]/255), *[s/h, t/w]] - mean_prev)
                        #print(dis)
                        if dis < bandwith:
                            is_visited[s][t] = 1
                            sum_dis[0] += img[s][t][0]/255
                            sum_dis[1] += img[s][t][1]/255
                            sum_dis[2] += img[s][t][2]/255
                            sum_dis[3] += s/h
                            sum_dis[4] += t/w
                            num_pts += 1
                #print(num_pts)
                mean = sum_dis / num_pts

                iteration += 1

                if np.linalg.norm(mean - mean_prev) < 0.05 * bandwith:
                    break
                else:
                    mean_prev = mean
        
            dis_pt_centroid = np.array([np.linalg.norm(mean - val) for val in centroid])


            if len(dis_pt_centroid) != 0 and np.amin(dis_pt_centroid) < bandwith:
                pass
            else:
                # create a new cluster
                centroid.append(mean)

            print(i, j)
            #print(centroid)

    print(centroid)
    for i in range(h):
        for j in range(w):
            dis_pt_centroid = np.array([np.linalg.norm(val - [*(img[i][j]/255), *[i/h, j/w]]) for val in centroid])
            centroid_id = np.argmin(dis_pt_centroid)
            #print(centroid_id)
            img[i][j] = centroid[centroid_id][0:3]*255
           
    cv2.imwrite(out_path, img)


def plt_RGBscatter(in_path, out_path):

    im = Image.open(in_path)
    px = im.load()

    ax = plt.axes(projection = '3d')
    x = []
    y = []
    z = []
    c = []

    for row in range(0,im.height):
        for col in range(0, im.width):
            pix = px[col,row]
            newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)

            if(not newCol in c):
                x.append(pix[0])
                y.append(pix[1])
                z.append(pix[2])
                c.append(newCol)
        
        print(row)

    ax.scatter(x, y, z, c = c)
    plt.savefig(out_path) 
    plt.clf()

def main(mode):

    print(mode)

    img = cv2.imread("./2/2-image.jpg")
    img2 = cv2.imread("./2/2-masterpiece.jpg")
    img_ = copy.deepcopy(img)
    img2_ = copy.deepcopy(img2)

    img = img.astype(np.int32)
    img2 = img2.astype(np.int32)

    if mode == "kmeans":
        #kmeans(img, k=4, out_path='./2/output/image/kmeans-4.jpg')
        kmeans(img, k=7, out_path='./2/output/image/kmeans-7.jpg')
        kmeans(img, k=11, out_path='./2/output/image/kmeans-11.jpg')
        kmeans(img2, k=4, out_path='./2/output/masterpiece/kmeans-4.jpg')
        kmeans(img2, k=7, out_path='./2/output/masterpiece/kmeans-7.jpg')
        kmeans(img2, k=11, out_path='./2/output/masterpiece/kmeans-11.jpg')

    elif mode == "kmeansPlusPlus":
        kmeansPlusPlus(img, k=4, out_path='./2/output/image/kmeansPlusPlus-4.jpg')
        kmeansPlusPlus(img, k=7, out_path='./2/output/image/kmeansPlusPlus-7.jpg')
        kmeansPlusPlus(img, k=11, out_path='./2/output/image/kmeansPlusPlus-11.jpg')
        kmeansPlusPlus(img2, k=4, out_path='./2/output/masterpiece/kmeansPlusPlus-4.jpg')
        kmeansPlusPlus(img2, k=7, out_path='./2/output/masterpiece/kmeansPlusPlus-7.jpg')
        kmeansPlusPlus(img2, k=11, out_path='./2/output/masterpiece/kmeansPlusPlus-11.jpg')


    elif mode == "MeanShift":
        s = (int(0.7*img_.shape[1]), int(0.7*img_.shape[0]))
        img_ = cv2.resize(img_, dsize=s, interpolation=cv2.INTER_AREA)
        img_ = img_.astype(np.int32)

        s2 = (int(0.7*img2_.shape[1]), int(0.7*img2_.shape[0]))
        img2_ = cv2.resize(img2_, dsize=s2, interpolation=cv2.INTER_AREA)
        img2_ = img2_.astype(np.int32)

        MeanShift(img, bandwith=20, out_path='./2/output/image/MeanShift-Bandwith20.jpg')
        MeanShift(img, bandwith=35, out_path='./2/output/image/MeanShift-Bandwith35.jpg')
        MeanShift(img, bandwith=50, out_path='./2/output/image/MeanShift-Bandwith50.jpg')
        MeanShift(img2, bandwith=20, out_path='./2/output/masterpiece/MeanShift-Bandwith20.jpg')
        MeanShift(img2, bandwith=35, out_path='./2/output/masterpiece/MeanShift-Bandwith35.jpg')
        MeanShift(img2, bandwith=50, out_path='./2/output/masterpiece/MeanShift-Bandwith50.jpg')

    elif mode == "MeanShift_with_Spatial":
        s = (int(0.75*img_.shape[1]), int(0.75*img_.shape[0]))
        img_ = cv2.resize(img_, dsize=s, interpolation=cv2.INTER_AREA)
        img_ = img_.astype(np.int32)

        s2 = (int(0.75*img2_.shape[1]), int(0.75*img2_.shape[0]))
        img2_ = cv2.resize(img2_, dsize=s2, interpolation=cv2.INTER_AREA)
        img2_ = img2_.astype(np.int32)

        MeanShift_Spatial_and_Color(img_, bandwith=0.15, out_path='./2/output/image/MeanShift-Spatial-Bandwith15.jpg')
        MeanShift_Spatial_and_Color(img_, bandwith=0.35, out_path='./2/output/image/MeanShift-Spatial-Bandwith35.jpg')
        MeanShift_Spatial_and_Color(img_, bandwith=0.50, out_path='./2/output/image/MeanShift-Spatial-Bandwith50.jpg')
        MeanShift_Spatial_and_Color(img2_, bandwith=0.15, out_path='./2/output/masterpiece/MeanShift-Spatial-Bandwith15.jpg')
        MeanShift_Spatial_and_Color(img2_, bandwith=0.35, out_path='./2/output/masterpiece/MeanShift-Spatial-Bandwith35.jpg')
        MeanShift_Spatial_and_Color(img2_, bandwith=0.50, out_path='./2/output/masterpiece/MeanShift-Spatial-Bandwith50.jpg')

    elif mode == "plt_scatter":
        plt_RGBscatter(in_path="./2/2-image.jpg", out_path="./2/output/image/RGB_Scatter_Before_MeanShift")
        plt_RGBscatter(in_path="./2/output/image/MeanShift-Bandwith20.jpg", out_path="./2/output/image/RGB_Scatter_After_MeanShift")
        plt_RGBscatter(in_path="./2/2-masterpiece.jpg", out_path="./2/output/masterpiece/RGB_Scatter_Before_MeanShift")
        plt_RGBscatter(in_path="./2/output/masterpiece/MeanShift-Bandwith20.jpg", out_path="./2/output/masterpiece/RGB_Scatter_After_MeanShift")

if __name__ == "__main__":
   main(sys.argv[1])