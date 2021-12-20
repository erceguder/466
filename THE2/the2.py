# Bugra Alparslan 2309607
# Erce Guder 2310068

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math


def parse(path: str):
    out_dir = None

    if path.endswith('/'):
        out_dir = "/".join(path.split('/')[:-1])
    else:
        out_dir = "/".join(path.split('/'))

    return out_dir + '/'


def distance(point1: tuple, point2: tuple):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def ideal_filter(img: np.ndarray, filter_type: str, r: float):
    rows, cols = img.shape
    center = (rows//2, cols//2)
    radius = r
        
    for r in range(rows):
        for c in range(cols):

            if (filter_type=='HIGH'):
                if (distance((r,c), center)) < radius:
                    img[r,c] = 0

            elif (filter_type=='LOW'):
                if (distance((r,c), center)) > radius:
                    img[r,c] = 0

    return img


def butterworth_filter(cutoff_freq: float):
    pass



def part1(input_img_path: str , output_path: str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    freq = np.fft.fft2(img)
    freq_shifted = np.fft.fftshift(freq)
    magnitude_spectrum = 20*np.log(np.abs(freq_shifted))

    filtered = ideal_filter(freq_shifted, 'HIGH', 200)
    filtered_inv_shift = np.fft.ifftshift(filtered)
    img_back = np.real(np.fft.ifft2(filtered_inv_shift))


    cv2.imwrite(f'./{out_dir}/edges.png', img_back)
        

if __name__ == "__main__":

    part1('THE2-Images/1.png', 'Outputs/EgdeDetection/')



