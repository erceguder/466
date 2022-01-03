# Bugra Alparslan 2309607
# Erce Guder 2310068

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from math import exp, sqrt

def parse(path: str):
    out_dir = None

    if path.endswith('/'):
        out_dir = "/".join(path.split('/')[:-1])
    else:
        out_dir = "/".join(path.split('/'))

    return out_dir + '/'


def distance(point1: tuple, point2: tuple):
    """
        euclidian distance between two points
    """
    return sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def ideal_filter(img: np.ndarray, filter_type: str, radius: float):
    rows, cols = img.shape
    center = (rows//2, cols//2)
        
    for r in range(rows):
        for c in range(cols):

            if filter_type == 'HIGH':
                if (distance((r,c), center)) < radius:
                    img[r,c] = 0

            elif filter_type == 'LOW':
                if (distance((r,c), center)) > radius:
                    img[r,c] = 0

    return img


def butterworth_high_pass(img_shape: tuple, D_0: int, n: int):
    """
        img_shape: shape of shifted frequency domain image
        D_0: cutoff frequency
        n: order
    """

    bwf = np.zeros(img_shape)

    rows = img_shape[0]
    cols = img_shape[1]

    center = (rows//2, cols//2)

    for u in range(rows):
        for v in range(cols):
            
            if (u, v) == center:
                bwf[u,v] = 0
            else:
                bwf[u,v] = 1/(1 + (D_0/distance(center, (u,v))) ** (2*n))

    return bwf


def gaussian_high_pass(img_shape: tuple, D_0: float):
    gaussian = np.zeros(img_shape)

    rows = img_shape[0]
    cols = img_shape[1]

    center = (rows // 2, cols // 2)

    for u in range(rows):
        for v in range(cols):

            gaussian[u,v] = 1 - exp(-distance((u,v), center)**2/(2*D_0*D_0))

    return gaussian


def part1(input_img_path: str , output_path: str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    freq = np.fft.fft2(img)
    freq_shifted = np.fft.fftshift(freq)

    cutoff = 100
    order = 2
    bw_filter = butterworth_high_pass(freq_shifted.shape, cutoff, order)

    filtered = np.multiply(bw_filter, freq_shifted)
    freq_inv_shift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(freq_inv_shift)
    img_back = np.real(img_back)

    cv2.imwrite(f'./{out_dir}/edges_cutoff{cutoff}_order{order}.png', img_back)


def enhance_3(path_to_3: str, output_path: str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass




def enhance_4(path_to_4: str, output_path: str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass


if __name__ == "__main__":

    # part1('THE2-Images/1.png', 'Outputs/EgdeDetection/')
    enhance_3('THE2-Images/3.png', 'Outputs/Enhance3/')


