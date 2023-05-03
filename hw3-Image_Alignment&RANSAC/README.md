cd ./HW3_108062373

## Execute Part 1
Problem 1-a and 1-b:
    => python3 1/HW3-1.py
    => output : SIFT-book{i}.jpg (i = 1~3) and RANSAC-book{i}.jpg (i = 1~3)

## Execute Part2

- Problem 2-a : 
    python3 2/HW3-2.py kmeans
    => output : kmeans-{i}.jpg (i = 4, 7, 11)

- Problem 2-b : 
    python3 2/HW3-2.py kmeansPlusPlus
    => output : kmeansPlusPlus-{i}.jpg (i = 4, 7, 11)

- Problem 2-c :  
    python3 2/HW3-2.py MeanShift 
    => output : MeanShift-Bandwidth{i}.jpg (i = 20, 35, 50)
    
    python3 2/HW3-2.py plt_scatter (需先執行前一行) 
    => output : RGB_Scatter_Before_MeanSift.jpg and RGB_Scatter_After_MeanSift.jpg

- Problem 2-d : 
    python3 2/HW3-2.py MeanShift_with_Spatial
    => output : MeanShift-Spatial-Bandwidth{i}.jpg (i = 15, 35, 50)

- Problem 2-e :
    執行完2-c與2-d以後，即會Output 2-e 的圖片
    => MeanShift-Bandwidth{i}.jpg (i = 20, 35, 50)
    => MeanShift-Spatial-Bandwidth{i}.jpg (i = 15, 35, 50)