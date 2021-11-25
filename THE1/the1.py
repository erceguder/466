import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def part1(input_img_path: str, output_path: str, m: list, s: list):
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    
    assert img is not None

    N = img.shape[0]*img.shape[1]
    out_dir = "/".join(output_path.split('/')[:-1])

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    # intensities = list()#[0 for i in range(256)]

    # for row in img:
    #     for element in row:
    #         intensities.append(element)#[element] += 1

    # plt.hist(intensities)
    # plt.savefig(out_dir + "/original_histogram.png")

    gaussians = np.ndarray(shape=(len(m), N))

    for i in range(len(m)):
        gaussians[i] = np.random.normal(loc=m[i], scale=s[i], size=N)
        plt.hist(gaussians[i], bins=100)
    
    plt.savefig(out_dir + "/gaussian_histogram.png")

if __name__ == "__main__":
    part1("THE1-Images/1.png", "Outputs/1/ex1.png", [45, 200], [45, 45])