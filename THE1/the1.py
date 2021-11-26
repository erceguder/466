import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def normalize(intensities: np.ndarray):
    g_m = np.subtract(intensities ,min(intensities))
    g_s = np.round(np.multiply(np.divide(g_m, max(g_m)), 255))

    return g_s.astype(int)

def find_closest(table: dict, s_k: int):
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

    plt.hist(gaussian, bins=50)
    plt.savefig(out_dir + "/gaussian_histogram.png")
    plt.clf()

    # Matching
    # unique, counts = np.unique(gaussian, return_counts=True)
    # print(dict(zip(unique, counts)))

    for intensity in gaussian:                                  # h of gaussians
        h_c_gaus[intensity] += 1

    for i in range(1, len(h_c_gaus)):                           # h_c of gaussians
        h_c_gaus[i] += h_c_gaus[i-1]

    # print("h_c_gaussian:")
    # print(h_c_gaus)

    T_1 = np.round(np.multiply((255./N), h_c_org))
    T_2 = np.round(np.multiply((255./(2*N)), h_c_gaus))

    table = dict(enumerate(T_2))
    z_k = [0 for _ in range(256)]

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
    
if __name__ == "__main__":
    ex = 1#input("Image: ")
    part1(f"THE1-Images/{ex}.png", f"Outputs/{ex}/ex{ex}.png", [45, 200], [45, 45])