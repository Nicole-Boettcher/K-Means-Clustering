import numpy as np
import math
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def init_centers(k, pixels, name):
    centers = np.zeros((k, 3))   # create array of size k
    if name == 'random':
        # init centers randomly from the range 
        for i in range(k):
            centers[i] = pixels[random.randint(0, pixels.shape[0])][random.randint(0, pixels.shape[1])]    # k centers with [r,g,b]

    elif name == 'large_distance':
        # init centers very far away from eachother
        # first center is randomly choosen 
        centers[0] = pixels[random.randint(0, pixels.shape[0])][random.randint(0, pixels.shape[1])]
        # for all pixels, compute the its distance to the previous center 

        for center_num in range(k-1):
            
            _, distances = calculate_distances(centers[:center_num+1], pixels)

            closest_center_index = np.argmin(distances, axis=2)    # find the center thats closest 

            closest_center_distance = np.empty((distances.shape[0], distances.shape[1]))
            for row in range(len(pixels)):
                for col in range(len(pixels[0])):
                    center_index = int(closest_center_index[row][col])
                    closest_center_distance[row][col] = distances[row][col][center_index]

            # pick pixel with largest distance - distances are [row, col, num_of_centers]
            largest_distance_index = np.unravel_index(np.argmax(closest_center_distance, axis=None), closest_center_distance.shape)   # should be [row, col]

            centers[center_num+1] = pixels[largest_distance_index[0]][largest_distance_index[1]]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], label='Starting center points', marker='o')
    # ax.set_xlabel('R-axis')
    # ax.set_ylabel('G-axis')
    # ax.set_zlabel('B-axis')

    # ax.legend()

    # plt.show()

    return centers

def calculate_distances(centers, pixels):

    distances = np.empty((pixels.shape[0], pixels.shape[1], len(centers)))
    assigned_class = np.zeros((pixels.shape[0], pixels.shape[1]))
    # iterate over all centers
    for row in range(len(pixels)):
        #print("row = ", row)
        for col in range(len(pixels[0])):
            for num_center in range(len(centers)):
                distances[row][col][num_center] = math.sqrt(math.pow(pixels[row][col][0] - centers[num_center][0], 2) + math.pow(pixels[row][col][1] - centers[num_center][1], 2) + math.pow(pixels[row][col][2] - centers[num_center][2], 2))
            assigned_class[row][col] = np.argmin(distances[row][col])
            #print(assigned_class[row][col])

    return assigned_class, distances

def new_centers(pixels, assigned_class, k_val):
    new_averages_r = np.zeros(k_val)
    new_averages_g = np.zeros(k_val)
    new_averages_b = np.zeros(k_val)
 
    for row in range(len(pixels)):
        for col in range(len(pixels[row])):
            new_averages_r[int(assigned_class[row][col])] += pixels[row][col][0] # for each pixel avergae the values and use the assigned class index to assign it to the correct center
            new_averages_g[int(assigned_class[row][col])] += pixels[row][col][1]
            new_averages_b[int(assigned_class[row][col])] += pixels[row][col][2]

    # divide by number of samples in each class 
    centers = np.zeros((k_val,3))   # init as empty is a problem becasue it gives garbage values for centers that dont have data points
    for k in range(k_val):
        class_count = np.count_nonzero(assigned_class == k)
        if class_count != 0:
            centers[k] = [new_averages_r[k] / class_count, new_averages_g[k] / class_count, new_averages_b[k] / class_count]
        else:
            print("no class ", k)
    return centers
            
def reconstruct_photo(assigned_class, pixels, centers):
    new_pixels = np.empty((pixels.shape))
    for row in range(len(pixels)):
        for col in range(len(pixels[row])):
            new_pixels[row][col] = centers[int(assigned_class[row][col])]

    return new_pixels

def image_loss(pixels, new_pixels):
    # mean squared error 
    # difference between each r b g - average them?
    pixels = pixels.astype(float)
    new_pixels = new_pixels.astype(float)

    mean_squared_error_sum = 0
    for row in range(len(pixels)):
        for col in range(len(pixels[0])):
                 #  + math.pow(pixels[row][col][1] - new_pixels[row][col][1], 2) + math.pow(pixels[row][col][2] - new_pixels[row][col][2], 2))
            red_diff = pixels[row][col][0] - new_pixels[row][col][0]
            green_diff = pixels[row][col][1] - new_pixels[row][col][1]
            blue_diff = pixels[row][col][2] - new_pixels[row][col][2]
            mean_squared_error_sum += (math.pow(red_diff, 2) + math.pow(green_diff, 2) + math.pow(blue_diff, 2))/3
        
    return mean_squared_error_sum / (len(pixels) * len(pixels[0]))

def main():
    position_graph = 1
    plt.subplot(2,3,position_graph)
    pixels = plt.imread("sky.jpg")
    plt.title("Original Image")
    plt.imshow(pixels)
    plt.plot()
    
    k_values = [2, 3, 10, 20, 40]
    for k_val in k_values:
        position_graph = position_graph + 1
        print("********* K = ", k_val, " ***********")
        pixels = plt.imread("sky.jpg").astype(float)
        init_strat = 'large_distance'
        center_diff = np.ones(k_val)
        converge = False
        num_iterations = 0
        centers = init_centers(k_val, pixels, init_strat)
        old_assigned_classes = np.empty((pixels.shape[0], pixels.shape[1]))
        print("init center values: ", centers)

        while converge == False:
        #while center_diff != 0:
            assigned_class, distances = calculate_distances(centers, pixels)
            centers = new_centers(pixels, assigned_class, k_val)
            print("Num of iterations = ", num_iterations)
            assigned_class_diff = np.subtract(assigned_class, old_assigned_classes)
            if (np.all(assigned_class_diff == 0)):
                #print("diff = ", assigned_class_diff)
                #print("all classes are the same")
                converge = True

            old_assigned_classes = assigned_class
            num_iterations = num_iterations + 1

        #print(assigned_class)
        new_pixels = np.round(reconstruct_photo(assigned_class, pixels, centers)).astype(np.uint8)

        mse = image_loss(pixels, new_pixels)
        print("Mean square error of image is ", mse)
        print("Number of iterations = ", num_iterations)
        plt.subplot(2,3,position_graph)
        title = "K = " +  str(k_val)
        plt.title(title)
        plt.imshow(new_pixels)
        plt.plot()
        
    plt.show()

main()