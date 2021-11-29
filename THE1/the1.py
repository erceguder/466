import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import median

def gaussians(m: list, s: list):
    def gaussian(x, mu, sigma):
        return np.exp( (((x - mu)/sigma)**2) / (-2) ) / (sigma * np.sqrt(2*np.pi))
    
    #return [[gaussian(x, mu, sigma) for x in np.arange(256)] for mu, sigma in zip(m, s)]
    mixture = None

    for mu, sigma in zip(m, s):
        if mixture == None:
            mixture = [gaussian(x, mu, sigma) for x in np.arange(256)]
        else:
            mixture = np.add(mixture, [gaussian(x, mu, sigma) for x in np.arange(256)])

    return np.divide(mixture, np.sum(mixture))

def find_closest(table: dict, s_k):
    diff = 255
    z_k = 0
    for i in table:
        if (abs(table[i] - s_k) < diff):
            diff = abs(table[i] - s_k)
            z_k = i
    return z_k

def parse(path: str):
    out_dir = None

    if path.endswith('/'):
        out_dir = "/".join(path.split('/')[:-1])
    else:
        out_dir = "/".join(path.split('/'))

    return out_dir + '/'

def part1(input_img_path: str, output_path: str, m: list, s: list):
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    
    assert img is not None

    N = img.shape[0]*img.shape[1]                               # number of pixels
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    intensities = list()
    h_c_org = np.zeros(256)
    h_c_gaus = np.zeros(256)

    for row in img:                                             # h of original image
        for intensity in row:
            intensities.append(intensity)
            h_c_org[intensity] += 1

    for i in range(1, len(h_c_org)):                            # h_c of original image
        h_c_org[i] += h_c_org[i-1]

    plt.hist(intensities, bins=50)
    plt.savefig(out_dir + "original_histogram.png")
    plt.clf()

    mixture = np.random.choice(256, N, True, gaussians(m, s))
    plt.hist(mixture, bins=50)
    plt.savefig(out_dir + "gaussian_histogram.png")
    plt.clf()
    
    # Matching
    for intensity in mixture:                                   # h of mixture
        h_c_gaus[intensity] += 1

    for i in range(1, len(h_c_gaus)):                           # h_c of mixture
        h_c_gaus[i] += h_c_gaus[i-1]

    T_1 = np.multiply((255./N), h_c_org)
    T_2 = np.multiply((255./N), h_c_gaus)

    table = dict(enumerate(T_2))
    z_k = np.zeros(256)

    for idx, s_k in enumerate(T_1):
        z_k[idx] = find_closest(table, s_k)

    intensities.clear()

    for row in img:
        for idx, intensity in enumerate(row):
            row[idx] = z_k[intensity]
            intensities.append(z_k[intensity])

    cv2.imwrite(out_dir + "matched_image.png", img)

    plt.hist(intensities, bins=50)
    plt.savefig(out_dir + "matched_image_histogram.png")
    
def the1_convolution(input_img_path:str, filter: list):
    """
        1. Is the image grayscale?
        2. is filter always 2D list?
    """
    # shape = (H, W, C)
    img = cv2.imread(input_img_path) # is it grayscale?

    assert img is not None

    img_height = img.shape[0]
    img_width = img.shape[1]

    kernel_height = len(filter)
    kernel_width = len(filter[0])

    output = np.ndarray(shape=(img_height - kernel_height +1, img_width - kernel_width + 1, 3))

    for row_idx in range(img_height - kernel_height +1):
        row_patch = img[row_idx: row_idx+kernel_height]
        
        for col_idx in range(img_width - kernel_width + 1):

            patch = row_patch[:, col_idx: col_idx+kernel_width]
            
            # print(patch.shape)
            # print(patch[:, :, 0].shape)
            # exit(0)


            output[row_idx][col_idx][0] = np.sum(np.multiply(patch[:, :, 0], filter))
            output[row_idx][col_idx][1] = np.sum(np.multiply(patch[:, :, 1], filter))
            output[row_idx][col_idx][2] = np.sum(np.multiply(patch[:, :, 2], filter))

    return output

def normalize(image:np.ndarray):
    g_m = np.subtract(image, image.min())
    g_s = np.round(np.multiply(np.divide(g_m, g_m.max()), 255))

    return g_s

def part2(input_img_path:str , output_path:str):
    """
        Applies Sobel edge detector on a grayscale image of path
        input_img_path and writes the edge map to output_path.
    """

    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    # create vertical and horizontal masks
    w_v = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    w_h = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    # get vertical and horizontal edge masks
    g_v = the1_convolution(input_img_path=input_img_path, filter=w_v)
    g_h = the1_convolution(input_img_path=input_img_path, filter=w_h)

    # get approximate gradient at each point by combining both edge masks
    g = np.sqrt(np.square(g_v) + np.square(g_h))
    g = normalize(g)

    cv2.imwrite(out_dir + "/edges.png", g)

def convolution(img, filter):

    img_height = img.shape[0]
    img_width = img.shape[1]

    kernel_height = len(filter)
    kernel_width = len(filter[0])

    output = np.ndarray(shape=(img_height - kernel_height +1, img_width - kernel_width + 1, 3))

    for row_idx in range(img_height - kernel_height +1):
        row_patch = img[row_idx: row_idx+kernel_height]
        
        for col_idx in range(img_width - kernel_width + 1):

            patch = row_patch[:, col_idx: col_idx+kernel_width]

            output[row_idx][col_idx][0] = np.sum(np.multiply(patch[:, :, 0], filter))
            output[row_idx][col_idx][1] = np.sum(np.multiply(patch[:, :, 1], filter))
            output[row_idx][col_idx][2] = np.sum(np.multiply(patch[:, :, 2], filter))

    return output

def median_filter(img, size):
    
    assert img is not None

    img_height = img.shape[0]
    img_width = img.shape[1]

    kernel_height = size
    kernel_width = size

    output = np.ndarray(shape=(img_height - kernel_height +1, img_width - kernel_width + 1, 3))

    for row_idx in range(img_height - kernel_height +1):
        row_patch = img[row_idx: row_idx+kernel_height]
        
        for col_idx in range(img_width - kernel_width + 1):

            patch = row_patch[:, col_idx: col_idx+kernel_width]
            
            output[row_idx][col_idx][0] = np.median(patch[:,:,0])
            output[row_idx][col_idx][1] = np.median(patch[:,:,1])
            output[row_idx][col_idx][2] = np.median(patch[:,:,2])

    return output

def enhance_3(path_to_3:str , output_path:str):

    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    gaussian_filters = [
        np.array([[1,2,1],[2,4,2],[1,2,1]]) * (1/16),
        np.array([[1,4,7,4,1], [4,16,26,16,4],[7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]]) * (1/273),
        np.array([[0,0,1,2,1,0,0], [0,3,13,22,13,3,0], [1,13,59,97,59,13,1], [2,22,97,159,97,22,2], [1,13,59,97,59,13,1], [0,3,13,22,13,3,0], [0,0,1,2,1,0,0]]) * (1/1003)
    ]

    laplacian_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])

    img = cv2.imread(path_to_3)

    output = median_filter(img, 21)
    cv2.imwrite(f"./{out_dir}/median21.png", output)

    # output = median_filter(img, 3)
    # output = convolution(output, laplacian_filter)
    # cv2.imwrite(f"./{out_dir}/median-laplace.png", output)

    # l1 = l2 = [3,5,7]

    # for i in l1:
    #     output1 = median_filter(img, i)
    #     for j in l2:
    #         output2 = convolution(output1, gaussian_filters[int((j-3)/2)])
    #         cv2.imwrite(f"./{out_dir}/median{i}-gaussian{j}.png", output2)

    # for i in l1:
    #     output1 = convolution(img, gaussian_filters[int((i-3)/2)])
    #     for j in l2:
    #         output2 = median_filter(output1, j)
    #         cv2.imwrite(f"./{out_dir}/gaussian{i}-median{j}.png", output2)

def enhance_4(path_to_4:str , output_path:str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass


    img = cv2.imread(path_to_4)
    laplacian_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    log_filter = np.array([[0,0,3,2,2,2,3,0,0],[0,2,3,5,5,5,3,2,0],[3,3,5,3,0,3,5,3,3],[2,5,3,-12,-23,-12,3,5,2],[2,5,0,-23,-40,-23,0,5,2],[2,5,3,-12,-23,-12,3,5,2],[3,3,5,3,0,3,5,3,3],[0,2,3,5,5,5,3,2,0],[0,0,3,2,2,2,3,0,0]])

    edge_mask = convolution(img, log_filter)
    # edge_mask = normalize(edge_mask)
    edge_mask[:,:,0] = normalize(edge_mask[:,:,0])
    edge_mask[:,:,1] = normalize(edge_mask[:,:,1])
    edge_mask[:,:,2] = normalize(edge_mask[:,:,2])

    median_filtered = median_filter(img, 7)
    smoothed = convolution(median_filtered, np.array([[1,2,1],[2,4,2],[1,2,1]]) * (1/16))
    output = smoothed - edge_mask

    output[:,:,0] = normalize(output[:,:,0])
    output[:,:,1] = normalize(output[:,:,1])
    output[:,:,2] = normalize(output[:,:,2])
    cv2.imwrite(f"./{out_dir}/bugrik2.png", output)

    # median_filtered = median_filter(img, 7)
    # median_smoothed = convolution(median_filtered, np.array([[1,2,1],[2,4,2],[1,2,1]]) * (1/16))

    # output = median_smoothed - edge_mask[3:-3, 3:-3, :]
    # output[:,:,0] = normalize(output[:,:,0])
    # output[:,:,1] = normalize(output[:,:,1])
    # output[:,:,2] = normalize(output[:,:,2])

    # cv2.imwrite(f"./{out_dir}/bugrik.png", output)

    smoothing_filter = [
                        [0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.1],
                        [0.1, 0.1, 0.1]
                    ]

    # print("initial shape = {}".format(img.shape))
    # median_filtered = median_filter(img, 3)
    # print("median filtered shape={}".format(median_filtered.shape))
    # median_smoothed = convolution(median_filtered, smoothing_filter)
    # print("median smoothed shape={}".format(median_smoothed.shape))
    # print(median_filtered[1:597, 1:591, :].shape)

    # diff = median_filtered[1:597, 1:591, :] - median_smoothed
    # output = median_filtered[1:597, 1:591, :] + diff
    # output = normalize(output)
    # output = np.multiply(output, 1.1)

    # cv2.imwrite(f"./{out_dir}/ercuk2.png", output)



    


if __name__ == "__main__":
    # ex = 2#input("Image: ")
    # part1(f"./THE1-Images/{ex}.png", "./Outputs/Part1/", [30, 230], [5, 5])
    # part2(f"./THE1-Images/{ex}.png", "./Outputs/Part2/")
    # enhance_3("./THE1-Images/3.png", "~")

    # box_filter = [  [1, 1, 1], 
    #                 [0, 0, 0],
    #                 [-1, -1, -1]]

    # simple_smoothing_filters = [
    #     np.array([[1,1,1],[1,1,1],[1,1,1]]) * (1/9),
    #     np.array([[1,1,1],[1,2,1],[1,1,1]]) * (1/10),
    #     np.array([[1,2,1],[1,4,1],[1,2,1]]) * (1/14)
    # ]

    # enhance_3("./THE1-Images/3.png", "Outputs/Enhance3/")
    enhance_4("./THE1-Images/4.png", "Outputs/Enhance4/")

    


    
