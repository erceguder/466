import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def normalize(intensities: np.ndarray):
    g_m = np.subtract(intensities ,min(intensities))
    g_s = np.round(np.multiply(np.divide(g_m, max(g_m)), 255))

    return g_s.astype(int)

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
    h_c_org = [0 for _ in range(256)]
    h_c_gaus = [0 for _ in range(256)]

    for row in img:                                             # h of original image
        for intensity in row:
            intensities.append(intensity)
            h_c_org[intensity] += 1

    for i in range(1, len(h_c_org)):                            # h_c of original image
        h_c_org[i] += h_c_org[i-1]

    plt.hist(intensities, bins=50)
    plt.savefig(out_dir + "/original_histogram.png")
    plt.clf()

    gaussian = np.random.normal(loc=m[0], scale=s[0], size=N)   # Mixture gaussian distributions

    for i in range(1, len(m)):
        gaussian = np.append(gaussian, np.random.normal(loc=m[i], scale=s[i], size=N))

    gaussian = normalize(gaussian)

    for intensity in gaussian:                                  # h of gaussians
        h_c_gaus[intensity] += 1

    for i in range(1, len(h_c_gaus)):                           # h_c of gaussians
        h_c_gaus[i] += h_c_gaus[i-1]

    plt.hist(gaussian, bins=50)
    plt.savefig(out_dir + "/gaussian_histogram.png")
    plt.clf()

    T_1 = np.multiply((255./N), h_c_org)
    T_2 = np.multiply((255./N), h_c_gaus)

    z_ks = [0 for _ in range(256)]

    for r_k, T_1_out in enumerate(T_1):
        for z_k, T_2_out in enumerate(T_2):
            
            if round(T_1_out) == round(T_2_out):
                z_ks[r_k] = z_k
                break
    
    intensities.clear()

    for row_idx,row in enumerate(img):
        for col_idx, intensity in enumerate(row):
            img[row_idx][col_idx] = z_ks[intensity]
            intensities.append(z_ks[intensity])

    cv2.imwrite(out_dir + "/matched_image.png", img)

    plt.hist(intensities, bins=50)
    plt.savefig(out_dir + "/matched_image_histogram.png")
    
if __name__ == "__main__":
    ex = input()
    part1(f"THE1-Images/{ex}.png", f"Outputs/{ex}/ex{ex}.png", [80, 160], [30, 30])