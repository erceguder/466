import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

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

def normalize(intensities: np.ndarray):
    g_m = np.subtract(intensities, min(intensities))
    g_s = np.round(np.multiply(np.divide(g_m, max(g_m)), 255))

    return g_s.astype(int)

def find_closest(table: dict, s_k):
    diff = 255
    z_k = 0
    for i in table:
        if (abs(table[i] - s_k) < diff):
            diff = abs(table[i] - s_k)
            z_k = i
    return z_k

def part1(input_img_path: str, output_path: str, m: list, s: list):
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    
    assert img is not None

    N = img.shape[0]*img.shape[1]                               # number of pixels
    out_dir = "/".join(output_path.split('/')[:-1])

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
    plt.savefig(out_dir + "/original_histogram.png")
    plt.clf()

    mixture = np.random.choice(256, N, True, gaussians(m, s))
    plt.hist(mixture, bins=50)
    plt.savefig(out_dir + "/gaussian_histogram.png")
    plt.clf()
    
    # Matching
    # unique, counts = np.unique(gaussian, return_counts=True)
    # print(dict(zip(unique, counts)))

    # gaussian = normalize(gaussian)

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

    cv2.imwrite(out_dir + "/matched_image.png", img)

    plt.hist(intensities, bins=50)
    plt.savefig(out_dir + "/matched_image_histogram.png")
    
def the1_convolution(input_img_path:str, filter: list):
    """
        1. Is the image grayscale?
        2. is filter always 2D list?
    """
    # shape = (H, W, C)
    img = cv2.imread(input_img_path, cv2.IMREAD_COLOR) # is it grayscale?

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

def part2(input_img_path , output_path):
    pass

if __name__ == "__main__":
    ex = 1#input("Image: ")
    part1(f"THE1-Images/{ex}.png", f"Outputs/{ex}/ex{ex}.png", [45, 200], [45, 10])

    # box_filter = [  [1, 1, 1], 
    #                 [0, 0, 0],
    #                 [-1, -1, -1]]

    # output = the1_convolution(f"THE1-Images/{ex}.png", box_filter)
    # cv2.imwrite("convolution.png", output)