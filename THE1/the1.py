# Bugra Alparslan 2309607
# Erce Guder 2310068

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

def the1_convolution(input_img_path:str, filter: list):
    """
        1. Is the image grayscale?
        2. is filter always 2D list?
    """
    # shape = (H, W, C)
    img = cv2.imread(input_img_path) # is it grayscale?
    
    assert img is not None

    return convolution(img, filter)

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

    img = cv2.imread(path_to_3)

    output = median_filter(img, 7)
    output = convolution(output, gaussian_filters[0])
    cv2.imwrite(f"./{out_dir}/enhance3.png", output)

    # below snippet produces every single combination of median and gaussian smoothing, for filter sizes of 3, 5 and 7.
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

def enhance_by_channel(img, out_dir):
    
    g = img[:,:,1]
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    g = median_filter(g, 3)
    img[1:-1,1:-1,1] = g[:,:,1]
    # cv2.imwrite(f"./{out_dir}/g_channel_median3.png", img)

    r = img[:,:,2]
    r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    r = median_filter(r,3)
    img[1:-1,1:-1,2] = r[:,:,2]
    # cv2.imwrite(f"./{out_dir}/g+r_channel_median3.png", img)

    b = img[:,:,0]
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    b = convolution(b, np.array([[1,2,1],[2,4,2],[1,2,1]]) * (1/16))
    img[1:-1,1:-1,0] = b[:,:,0]
    cv2.imwrite(f"./{out_dir}/enhance4.png", img)

def enhance_4(path_to_4:str , output_path:str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass
    
    img = cv2.imread(path_to_4)
    
    # Applies median filter to R and G channels, smooths B channel
    enhance_by_channel(img, out_dir)
    


if __name__ == "__main__":

    # enhance_3("./THE1-Images/3.png", "Outputs/Enhance3/")
    # enhance_4("./THE1-Images/4.png", "Outputs/Enhance4/")

    


    
