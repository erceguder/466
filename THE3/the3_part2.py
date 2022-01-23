import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import os


def parse(path: str):
    out_dir = None

    if path.endswith('/'):
        out_dir = "/".join(path.split('/')[:-1])
    else:
        out_dir = "/".join(path.split('/'))

    return out_dir + '/'


def segmentation_function_ncut():
    out_dir = parse('./Outputs/')
    pass


def make_output(n, out_dir):
   
    if n == 1:
        img = cv2.imread(f'./THE3-Images/B{n}.png')
        index = pd.MultiIndex.from_product((*map(range, img.shape[:2]), ('b', 'g', 'r')), names=('row', 'col', None))

        df1 = pd.Series(img.flatten(), index=index)
        df1 = df1.unstack()
        df1 = df1.reset_index().reindex(columns=['col','row', 'b','g','r'])

        df1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(df1)

        # bw1 = estimate_bandwidth(df1, quantile=.04, n_jobs=-1)
        bw1 = 0.25242707906633016
        mult = 1.75
        bw1 *= mult
        ms1 = MeanShift(bandwidth=bw1, bin_seeding=True, cluster_all=True).fit(df1)

        res_img = np.reshape(ms1.labels_, img.shape[:2])

        plt.imsave(f'./{out_dir}/the3_B{n}_output.png', res_img)

    elif n == 2:
        img = cv2.imread(f'./THE3-Images/B{n}.png')
        index = pd.MultiIndex.from_product((*map(range, img.shape[:2]), ('b', 'g', 'r')), names=('row', 'col', None))

        df1 = pd.Series(img.flatten(), index=index)
        df1 = df1.unstack()
        df1 = df1.reset_index().reindex(columns=['col','row', 'b','g','r'])

        df1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(df1)
        
        bw1 = 0.23870906024966218
        mult = 1.70
        bw1 *= mult
        ms1 = MeanShift(bandwidth=bw1, bin_seeding=True, cluster_all=True).fit(df1)

        res_img = np.reshape(ms1.labels_, img.shape[:2])

        plt.imsave(f'./{out_dir}/the3_B{n}_output.png', res_img)

    elif n == 3:
        img = cv2.imread(f'./THE3-Images/B{n}.png')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img,(13,13),0)
        index = pd.MultiIndex.from_product((*map(range, img.shape[:2]), ('b', 'g', 'r')), names=('row', 'col', None))

        df1 = pd.Series(img.flatten(), index=index)
        df1 = df1.unstack()
        df1 = df1.reset_index().reindex(columns=['col','row', 'b','g','r'])

        col_list = list(df1)
        col_list.remove('col')
        col_list.remove('row')
        df4 = df1[col_list].sum(axis=1).to_frame()

        df4 = MinMaxScaler(feature_range=(0, 1)).fit_transform(df4)

        # bw4 = estimate_bandwidth(df4, quantile=.04, n_jobs=-1)
        bw4 = 0.018754987319381165
        mult = 5.5
        bw4 *= mult
        ms4 = MeanShift(bandwidth=bw4, bin_seeding=True, cluster_all=True).fit(df4)

        res_img = np.reshape(ms4.labels_, img.shape[:2])

        plt.imsave(f'./{out_dir}/the3_B{n}_output.png', res_img)



def segmentation_function_meanshift():
    
    out_dir = parse('./Outputs/')

    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    for i in range(1,4):
        make_output(i, out_dir)



if __name__ == '__main__':
    
    segmentation_function_meanshift()
