# Bugra Alparslan 2309607
# Erce Guder 2310068

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from math import exp, sqrt
import scipy.fftpack as fftpack
import json

lum_quant_table =  [[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]]

chrom_quant_table = [[17, 18, 24, 47, 99, 99, 99, 99],
                     [18, 21, 26, 66, 99, 99, 99, 99],
                     [24, 26, 56, 99, 99, 99, 99, 99],
                     [47, 66, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99],
                     [99, 99, 99, 99, 99, 99, 99, 99]]

zigzag_indices = None
from math import exp, sqrt, e

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

    cv2.imwrite(f'./{out_dir}/edges.png', img_back)


def median_filter(img, size):
    assert img is not None

    img_height = img.shape[0]
    img_width = img.shape[1]

    kernel_height = size
    kernel_width = size

    output = img

    for row_idx in range(img_height-2):
        row_patch = img[row_idx: row_idx + kernel_height]

        for col_idx in range(img_width-2):
            patch = row_patch[:, col_idx: col_idx + kernel_width]

            output[row_idx+1][col_idx+1] = np.median(patch)

    return output


def enhance_3(path_to_3: str, output_path: str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    img = cv2.imread(path_to_3, cv2.IMREAD_COLOR)

    gray_r = img[:, :, 2]
    gray_g = img[:, :, 1]
    gray_b = img[:, :, 0]

    freq_r, freq_g, freq_b = np.fft.fft2(gray_r), np.fft.fft2(gray_g), np.fft.fft2(gray_b) 
    fshift_r, fshift_g, fshift_b = np.fft.fftshift(freq_r), np.fft.fftshift(freq_g), np.fft.fftshift(freq_b)
    magnitude_r, magnitude_g, magnitude_b = 1+np.log(np.abs(fshift_r)), 1+np.log(np.abs(fshift_g)), 1+np.log(np.abs(fshift_b))
    angle_r, angle_g, angle_b = np.angle(fshift_r), np.angle(fshift_g), np.angle(fshift_b)

    center_r = (magnitude_r.shape[0] // 2, magnitude_r.shape[1] // 2)
    center_g = (magnitude_g.shape[0] // 2, magnitude_g.shape[1] // 2)
    center_b = (magnitude_b.shape[0] // 2, magnitude_b.shape[1] // 2)

    magnitude_r[center_r[0]-100:center_r[0]+100, center_r[1]-100:center_r[1]+100] = median_filter(magnitude_r[center_r[0]-100:center_r[0]+100, center_r[1]-100:center_r[1]+100], 7)
    magnitude_g[center_g[0] - 100:center_g[0] + 100, center_g[1] - 100:center_g[1] + 100] = median_filter(
        magnitude_g[center_g[0] - 100:center_g[0] + 100, center_g[1] - 100:center_g[1] + 100], 7)
    magnitude_b[center_b[0] - 100:center_b[0] + 100, center_b[1] - 100:center_b[1] + 100] = median_filter(
        magnitude_b[center_b[0] - 100:center_b[0] + 100, center_b[1] - 100:center_b[1] + 100], 7)

    row = angle_r.shape[0]
    col = angle_r.shape[1]

    complex_angle_r, complex_angle_g, complex_angle_b = np.empty(angle_r.shape, dtype=complex), np.empty(angle_g.shape, dtype=complex), np.empty(angle_b.shape, dtype=complex)
    for i in range(row):
        for j in range(col):
            complex_angle_r[i,j] = complex(0, angle_r[i,j])
            complex_angle_g[i,j] = complex(0, angle_g[i,j])
            complex_angle_b[i,j] = complex(0, angle_b[i,j])

    f_r = np.multiply(np.exp(np.subtract(magnitude_r, 1)), np.power(e, complex_angle_r))
    f_g = np.multiply(np.exp(np.subtract(magnitude_g, 1)), np.power(e, complex_angle_g))
    f_b = np.multiply(np.exp(np.subtract(magnitude_b, 1)), np.power(e, complex_angle_b))

    back_r, back_g, back_b = np.real(np.fft.ifft2(np.fft.ifftshift(f_r))), np.real(np.fft.ifft2(np.fft.ifftshift(f_g))), np.real(np.fft.ifft2(np.fft.ifftshift(f_b)))

    back_r, back_g, back_b = back_r.astype(np.uint8), back_g.astype(np.uint8), back_b.astype(np.uint8)
    back_r = cv2.normalize(back_r, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    back_g = cv2.normalize(back_g, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    back_b = cv2.normalize(back_b, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    back_rgb = np.dstack((back_r, back_g, back_b))

    plt.imshow(back_rgb)
    plt.show()



def enhance_4(path_to_4: str, output_path: str):
    
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

def dct(img):
    return fftpack.dct(fftpack.dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")

def idct(img):
    return fftpack.idct(fftpack.idct(img, axis=0, norm="ortho"), axis=1, norm="ortho")

def huffman_encode(patch: np.ndarray):
    class Node:
        def __init__(self, freq, val=None, left=None, right=None):
            self.freq = freq
            self.val = val
            self.left = left
            self.right = right
            self.code = ''

    def code_tree(codes: dict, tree: Node, code: str):
        if tree.left is not None:   # right is also not None
            tree.left.code = code + '0'
            code_tree(codes, tree.left, tree.left.code)

            tree.right.code = code + '1'
            code_tree(codes, tree.right, tree.right.code)
        else:
            codes[tree.val] = code

    nodes = []

    unique, counts = np.unique(patch, return_counts=True)
    dic = dict(zip(unique, counts))

    for key in dic:
        nodes.append(Node(dic[key], int(key)))

    while len(nodes) != 1:
        nodes.sort(key=lambda x: x.freq)

        node1 = nodes.pop(0)
        node2 = nodes.pop(0)

        nodes.append(Node(node1.freq + node2.freq, None, node1, node2))

    codes = dict()
    code_tree(codes, nodes[0], '')

    # in case the patch consists of all zeroes
    if len(codes) == 1:
        for symbol in codes:
            codes[symbol] = '0'

    return codes

def huffman_decode(string, dic):
    start, end = 0, 1
    result = list()

    while start != (len(string)):
            substr = string[start: end]
            found = False

            for key in dic:
                    if dic[key] == substr:
                            found = True
                            start = end
                            result.append(int(key))
            if not found:
                    end += 1

    return result

def zigzag(patch: np.ndarray):
    global zigzag_indices

    if zigzag_indices is None:
        zigzag_indices = list()
        for i in range(1, 16):
            for j in range(i):
                if i % 2 == 1:
                    zigzag_indices.append((i-1-j, j))
                else:
                    zigzag_indices.append((j, i-1-j))

        zigzag_indices = list(filter(lambda x: x[0] < 8 and x[1] < 8, zigzag_indices))

    result = list()

    for index in zigzag_indices:
        result.append(patch[index[0]][index[1]])

    return result

def reverse_zigzag(vals):
    global zigzag_indices

    if zigzag_indices is None:
        zigzag_indices = list()
        for i in range(1, 16):
            for j in range(i):
                if i % 2 == 1:
                    zigzag_indices.append((i-1-j, j))
                else:
                    zigzag_indices.append((j, i-1-j))

        zigzag_indices = list(filter(lambda x: x[0] < 8 and x[1] < 8, zigzag_indices))

    result = np.zeros((8, 8))

    for i in range(64):
        result[zigzag_indices[i][0]][zigzag_indices[i][1]] = vals[i]

    return result

def run_len_encode(string):

    element = string[0]
    result = str()#list()
    repeat = 1

    for char in string[1:]:
        if char == element:
            repeat += 1
        else:
            #result.append((int(element), repeat))
            result += element + str(repeat) + "-"
            repeat = 1
            element = char

    # result.append((int(element), repeat))
    result += element + str(repeat)

    return result

def run_len_decode(string):
    # 01-14-05-12-01-13-03-158
    symbols = string.split('-')
    decoded = str()

    for symbol in symbols:
        decoded += int(symbol[1:]) * symbol[0]

    return decoded

def the2_write(input_img_path: str, output_path: str):
    out_dir = parse(output_path)

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    img_name = out_dir + input_img_path.split('/')[-1].split('.')[0] + ".json"

    f = open(img_name, "w")

    bgr_img = cv2.imread(input_img_path)
    ycbcr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    shape = ycbcr_img.shape

    transformed = np.zeros(shape)
    out = dict()

    out["lum_quant_table"] = lum_quant_table
    out["chrom_quant_table"] = chrom_quant_table
    out["shape"] = shape

    # DCT and quantization
    for channel in range(3):
        for i in range(shape[0] // 8):  # Assuming the spatial resolution is multiple of 8.
            for j in range(shape[1] // 8):
                patch = transformed[8*i: 8*(i+1), 8*j:8*(j+1), channel] = dct(ycbcr_img[8*i: 8*(i+1), 8*j:8*(j+1), channel])

                if channel != 0:    # chroma channel
                    transformed[8*i: 8*(i+1), 8*j:8*(j+1), channel] = np.divide(patch, chrom_quant_table).astype(int)

                else:               # lum channel
                    transformed[8*i: 8*(i+1), 8*j:8*(j+1), channel] = np.divide(patch, lum_quant_table).astype(int)

    transformed = transformed.astype(int)

    for channel in range(3):
        huff_codes = huffman_encode(transformed[:, :, channel])
        out["huffman_" + str(channel)] = huff_codes

        out[channel] = {}

        for i in range(shape[0] // 8):  # Assuming the spatial resolution is multiple of 8.

            out[channel][i] = {}

            for j in range(shape[1] // 8):

                patch = transformed[8*i: 8*(i+1), 8*j:8*(j+1), channel]
                ordered = zigzag(patch)

                huff_encoded = str()
                for element in ordered:
                    huff_encoded += huff_codes[element]

                out[channel][i][j] = run_len_encode(huff_encoded)

    json.dump(out, f, separators=(',', ':'))
    f.close()

    return img_name

def the2_read(input_img_path: str):
    f = open(input_img_path, "r")

    img = json.load(f)

    lum_quant_table = img["lum_quant_table"]
    chrom_quant_table = img["chrom_quant_table"]
    shape = img["shape"]

    reconstructed_img = np.zeros((shape[0], shape[1], shape[2]))

    for channel in range(3):
        huff_codes = img["huffman_" + str(channel)]

        for i in range(shape[0] // 8):
            for j in range(shape[1] // 8):
                run_len_decoded = run_len_decode(img[str(channel)][str(i)][str(j)])
                huff_decoded = huffman_decode(run_len_decoded, huff_codes)
                # reconstructed_img[8*i:8*(i+1), 8*j:8*(j+1), channel] = reverse_zigzag(huff_decoded)

                if channel != 0:
                    reconstructed_img[8*i:8*(i+1), 8*j:8*(j+1), channel] = idct(np.multiply(reverse_zigzag(huff_decoded), chrom_quant_table))
                else:
                    reconstructed_img[8*i:8*(i+1), 8*j:8*(j+1), channel] = idct(np.multiply(reverse_zigzag(huff_decoded), lum_quant_table))

    plt.figure()
    plt.imshow(cv2.cvtColor(np.float32(reconstructed_img/255.), cv2.COLOR_YCrCb2RGB))
    plt.show()

    f.close()

if __name__ == "__main__":

    # part1('THE2-Images/1.png', 'Outputs/EgdeDetection/')
    # enhance_3('THE2-Images/3.png', 'Outputs/Enhance3/')

    the2_read(the2_write("THE2-Images/5.png", "outputs/"))
