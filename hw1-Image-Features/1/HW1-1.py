import math
import numpy as np
import cv2
import copy
from scipy.signal import convolve2d
import sys

def GaussianSmooth(img, kernel_size, sigma):
    '''
    input :
        img : input an image with np array
        sigma : parameter for Guassian Function
        kernel_size : size of kernel(filter)
    
    output:
        out : return an image that applys Guassain Filter
    '''
    x_center = math.floor(kernel_size/2)
    y_center = math.floor(kernel_size/2)

    kernel = np.zeros((kernel_size, kernel_size))

    # initialize the Gaussian Filter
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = j - x_center
            y = i - y_center
            kernel[i][j] = math.exp((-(x**2 + y**2))/(2 * (sigma**2))) / (2 * math.pi * sigma**2)

    # convolution the input image with Gaussian Kernel(Filter) => Gaussian Smoothing
    out = np.zeros(img.shape)
    out = np.round(convolve2d(img, kernel, boundary='symm', mode='same'))

    return out

def ImageGradient(img):
    '''
    input:
        img : an "blured" gray image
    
    output:
        gradient_magnitude : the gradient magnitude of every pixel in image
        gradient_direction : the gradient direction of every pixel in image
    '''

    # Sobel Operator(Kernel)
    Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #/ 8.
    Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) #/ 8.
    
    # Apply sobel operator to the image => Thus we can get the Gradient of an image
    gradient_x = np.round(convolve2d(img, Hx, boundary='symm', mode='same'))
    gradient_y = np.round(convolve2d(img, Hy, boundary='symm', mode='same'))

    gradient_magnitude = np.zeros(gradient_x.shape)
    gradient_direction = np.zeros(gradient_x.shape)

    # Calculate the gradient magnitude and direction of evey pixel in image
    for i in range(gradient_x.shape[0]):
        for j in range(gradient_x[i].shape[0]):
            gradient_magnitude[i][j] = (math.sqrt(gradient_x[i][j]**2 + gradient_y[i][j]**2))
            if gradient_x[i][j] != 0:
                gradient_direction[i][j] = math.atan(gradient_y[i][j]/gradient_x[i][j])
            else:
                # if gradient_x[i][j] == 0 => gradient_x[i][j] / gradient_y[i][j] = +inf or -inf
                if gradient_y[i][j] > 0:
                    gradient_direction[i][j] = math.pi /2
                else:
                    gradient_direction[i][j] = -math.pi /2
            if(gradient_magnitude[i][j] < 7):
                gradient_magnitude[i][j] = 0
                gradient_direction[i][j] = 0

    return gradient_magnitude, gradient_direction

def getGradientDirImg(gradient_direction, mag):
    
    angle = gradient_direction * 180. / np.pi
    
    gradientDirImg = np.zeros((gradient_direction.shape[0], gradient_direction.shape[1], 3))
    

    for i in range(0, gradientDirImg.shape[0]):
        for j in range(0, gradientDirImg.shape[1]):
            if mag[i][j] > 0.2:
                if (0 <= angle[i][j] < 22.5):
                    gradientDirImg[i][j] = [0, 20, 241]
                elif (22.5 <= angle[i][j] < 67.5):
                    gradientDirImg[i][j] = [17, 223, 20]
                elif (67.5 <= angle[i][j] < 112.5):
                    gradientDirImg[i][j] = [209, 2, 225]
                elif (112.5 <= angle[i][j] < 157.5):
                    gradientDirImg[i][j] = [20, 2, 225]
                elif (157.5 < angle[i][j] <= 180):
                    gradientDirImg[i][j] = [250, 75, 100]
                elif (-22.5 <= angle[i][j] < 0):
                    gradientDirImg[i][j] = [206, 206, 0]
                elif (-67.5 <= angle[i][j] < -22.5):
                    gradientDirImg[i][j] = [253, 223, 170]
                elif (-112.5 <= angle[i][j] < -67.5):
                    gradientDirImg[i][j] = [100, 250, 125]
                elif (-157.5 <= angle[i][j] < -112.5):
                    gradientDirImg[i][j] = [200, 100, 25]
                elif (-180 <= angle[i][j] < -157.5):
                    gradientDirImg[i][j] = [50, 0, 220]

    return gradientDirImg


def StructureTensor(img, win_size):
    '''
    Input:
        img: GrayScale Input image
        win_size: The window size of the local Stucture Matrix
    Output:
        R: The matrix that stores the response of every pixel in image

    Implementation of Corner and edge detection
    First, Calculate the local Structure Matrix for evey pixel in image,
    Second, Calculate the Response (wheter the pixel is flat, edge, or Corner) 
    of every pixel in img through the local structure Matrix.
    
    Note : The Respose Function = det(A) - K(Trace(A)**2), where A is a local
    Structure Matrix of a pixel in an image. where k = 0.04~0.06 (by experiment)
    if Resonse < 0 && Resonse < Threshold, the pixel is an edge in the image
    elif Resonse > 0 && Resonse < Threshold, the pixel is an Corner in the image
    else the pixel is a flat area in the image
    '''
    # sobel operator
    Hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.
    Hy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8.

    # Calculate the Gradient in both x and y  direction
    Ix = (convolve2d(img, Hx, boundary='symm', mode='same'))
    Iy = (convolve2d(img, Hy, boundary='symm', mode='same'))

    # Padding
    Ix_pad = np.zeros((Ix.shape[0]+int(round(win_size/2)), Ix.shape[1]+int(round(win_size/2))))
    Ix_pad[int(round(win_size/2)/2):Ix.shape[0]+int(round(win_size/2)/2), int(round(win_size/2)/2):Ix.shape[1]+int(round(win_size/2)/2)] = Ix
    Iy_pad = np.zeros((Iy.shape[0]+2, Iy.shape[1]+2))
    Iy_pad[int(round(win_size/2)/2):Iy.shape[0]+int(round(win_size/2)/2), int(round(win_size/2)/2):Iy.shape[1]+int(round(win_size/2)/2)] = Iy
    
    
    # round the Gradient values in both x and y direction to integer
    for i in range(Ix_pad.shape[0]):
        for j in range(Ix_pad.shape[1]):
            Ix_pad[i][j] = int(round(Ix_pad[i][j]))
    Ix_pad = Ix_pad.astype(int)
    
    for i in range(Iy_pad.shape[0]):
        for j in range(Iy_pad.shape[1]):
            Iy_pad[i][j] = int(round(Iy_pad[i][j]))
    Iy_pad = Iy_pad.astype(int)

    # R stores the output Response function (which was already mentioned above)
    # of evey pixel in the image.
    R = []

    k = 0.04

    # Calculate the Response Value of every pixel in the image
    for i in range(img.shape[0]):
        l = []
        for j in range(img.shape[1]):
            IxIx = convolve2d(Ix_pad[i:i+win_size, j:j+win_size], Ix_pad[i:i+win_size, j:j+win_size], mode='valid')[0][0]
            IxIy = convolve2d(Ix_pad[i:i+win_size, j:j+win_size], Iy_pad[i:i+win_size, j:j+win_size], mode='valid')[0][0]
            IyIy = convolve2d(Iy_pad[i:i+win_size, j:j+win_size], Iy_pad[i:i+win_size, j:j+win_size], mode='valid')[0][0]
            h = [[IxIx, IxIy], [IxIy, IyIy]]
    
            sign, logdet = np.linalg.slogdet(h)
            determinate = np.exp(logdet)
            Trace = (np.trace(h))
            r = determinate - k * (Trace ** 2)
            l.append(r)

        R.append(l)

    return R

def GetEdgeImage(img, R, Threshold=-80):
    Img = np.zeros(img.shape)
    for i, val in enumerate(Img):
        for j, k in enumerate(val):
            if R[i][j] < Threshold:
                Img[i, j, 0] = 255
                Img[i, j, 1] = 255
                Img[i, j, 2] = 255
            else:
                Img[i, j, 0] = 0
                Img[i, j, 1] = 0
                Img[i, j, 2] = 0
    return Img

def NonMaximalSuppression(img, gradient_direction, R, threshold=5000):
    '''
    input:
        img: the GrayScale Image
        gradient_direction: the direction of gradient of every pixel in the image (return from ImageGradient Function)
        R: The Response Matrix (return from StructureTensor Function)
        threshold: to determine the threshold of flat region and corner for response values in response function
    output:
        Corner_img: the image which visualize the corners

    Implementation of Non-Maximum Suppresion algorithm to find local maximum of local Structure Tensor

    How to determine wheter the pixel in image is local maximum?
    Check Wheter the Response Value of the pixel is larger than the Response Values of it's neighboor 
    "along the Gradient Direction".
    '''
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    R = np.array(R)
    Corner = np.zeros(R.shape)
    Corner_img = np.zeros(R.shape)

    for i in range(1, R.shape[0]-1):
        for j in range(1, R.shape[1]-1):
            if R[i][j] > threshold:
                if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                    # angle 0 and 180
                    if R[i][j] >= R[i][j-1] and R[i][j] >= R[i][j+1]:
                        Corner[i][j] = 1
                elif (22.5 <= angle[i][j] < 67.5):
                    # angle 45
                    if R[i][j] >= R[i-1][j-1] and R[i][j] >= R[i+1][j+1]:
                        Corner[i][j] = 1
                elif (67.5 <= angle[i][j] < 112.5):
                    # angle 90
                    if R[i][j] >= R[i-1][j] and R[i][j] >= R[i+1][j]:
                        Corner[i][j] = 1
                else:
                    # angle 135
                    if R[i][j] >= R[i+1][j-1] and R[i][j] >= R[i-1][j+1]:
                        Corner[i][j] = 1

    Corner_img = Corner * 255

    return Corner_img


def main(argv):

    file_name = argv[0]
    saved_file = ''
    if len(argv) == 1:
        Scale_and_Rotate = False
    else:
        if argv[1] == 'True':
            Scale_and_Rotate = True
        else:
            Scale_and_Rotate = False
    
    print(argv[0])
    print(argv[1])

    BGRimage = cv2.imread('./1/' + file_name + '.jpg')
    Grayimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2GRAY)

    if Scale_and_Rotate == True:
        h = Grayimage.shape[0]
        w = Grayimage.shape[1]

        M = cv2.getRotationMatrix2D((w//2, h//2), 30, 0.5)
        Grayimage = cv2.warpAffine(Grayimage, M, (w,h))
        saved_file = 'transformed/' + file_name
    else:
        saved_file = 'normal/' + file_name

    # Q1A-a
    out = GaussianSmooth(Grayimage, kernel_size=5, sigma=5)
    out1 = GaussianSmooth(Grayimage, kernel_size=10, sigma=5)

    cv2.imwrite('./1/output/' + saved_file + '_a_kernel5.jpg', out)
    cv2.imwrite('./1/output/' + saved_file + '_a_kernel10.jpg', out1)


    # Q1A-b
    gradient_magnitude, gradient_direction = ImageGradient(img=out)
    gradientDir_img = getGradientDirImg(gradient_direction,gradient_magnitude)
    gradient_magnitude1, gradient_direction1 = ImageGradient(img=out1)
    gradientDir_img1 = getGradientDirImg(gradient_direction1,gradient_magnitude)
    cv2.imwrite('./1/output/' + saved_file + '_b_magnitude_kernel5.jpg', gradient_magnitude)
    cv2.imwrite('./1/output/' + saved_file + '_b_magnitude_kernel10.jpg', gradient_magnitude1)
    cv2.imwrite('./1/output/' + saved_file + '_b_direction_kernel5.jpg', gradientDir_img)
    cv2.imwrite('./1/output/' + saved_file + '_b_direction_kernel10.jpg', gradientDir_img1)

    
    # Q1A-c
    R1 = StructureTensor(out1, 3)
    Img = GetEdgeImage(BGRimage, R1)
    cv2.imwrite('./1/output/' + saved_file + '_c_HarrisEdge_windowsize_3.jpg', Img)

    R2 = StructureTensor(out1, 5)
    Img = GetEdgeImage(BGRimage, R2)
    cv2.imwrite('./1/output/' + saved_file + '_c_HarrisEdge_windowsize_5.jpg', Img)


    #Q1A-d
    CornerImg1 = NonMaximalSuppression(Grayimage, gradient_direction1, R1, 500)
    cv2.imwrite('./1/output/' + saved_file + '_d_Corner_windowsize_3.jpg', CornerImg1)

    CornerImg2 = NonMaximalSuppression(Grayimage, gradient_direction1, R2, 5000)
    cv2.imwrite('./1/output/' + saved_file + '_d_Corner_windowsize_5.jpg', CornerImg2)
    

if __name__ == "__main__":
   main(sys.argv[1:])