#-*-coding: utf-8 -*-
import numpy as np
import skimage.io as sk
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import classification_report, precision_score, cohen_kappa_score
import time
import TemporelStability as TS
from tqdm import tqdm 
import tools
import cv2

def browseVideos(path):
    files = []
    for (dirpath, dirname, filesname) in os.walk(path):
        #print("=====> ", dirpath, dirname)
        for dir_n in dirname:
            for (dir_path, dir_name, filesname) in os.walk(os.path.join(path, dir_n)):
                for fname in filesname:
                    #if filename.endswith(extension):
                    #print(os.path.join(os.path.join(path, dir_n), fname))
                    files.append(os.path.join(os.path.join(path, dir_n), fname))
                
    return files

def compute_TS(path_dataset, result_path, n_clusters=4, miniBtach=True):
    files = browseVideos(path_dataset)
    print("nb videos is ", len(files))
    
    os.makedirs(result_path + "TS/", exist_ok=True)
    os.makedirs(result_path + "TS_t/", exist_ok=True)
    os.makedirs(result_path + "TS_s/", exist_ok=True)
    os.makedirs(result_path + "TS_st/", exist_ok=True)

    for file in tqdm(files):
        path, video_name = os.path.split(file)
        path, folder = os.path.split(path)
        video_name = os.path.splitext(video_name)[0]


        #print(folder, video_name)
        os.makedirs(result_path + "TS/" + folder + "/", exist_ok=True)
        os.makedirs(result_path + "TS_t/" + folder + "/", exist_ok=True)
        os.makedirs(result_path + "TS_s/" + folder + "/", exist_ok=True)
        os.makedirs(result_path + "TS_st/" + folder + "/", exist_ok=True)

        
        frames = tools.readVideo(file, size_ratio=None, color=False)#.astype(np.float) #/ 255.
        #print("Shape of frames: ", frames.shape)
        if frames.ndim == 4:
            frames = frames.transpose((1,2,3,0))
        else:
            frames = frames.transpose((1,2,0))
        

        st = TS.TemporelStability(n_clusters=n_clusters, miniBtach=miniBtach)
        date_vec = np.arange(frames.shape[-1])

        st.fit(frames)

        ts = st.get_TS(date_vec)
        ts_t = st.get_TS_t(date_vec)
        ts_s = st.get_TS_s(date_vec)
        ts_st = st.get_TS_st(date_vec)

        #  Normalization of the extracted features to visualization
        ts = (ts - np.min(ts, axis=(0,1))) / (np.max(ts, axis=(0,1)) - np.min(ts, axis=(0,1))) * 255
        ts_t = (ts_t - np.min(ts_t, axis=(0,1))) / (np.max(ts_t, axis=(0,1)) - np.min(ts_t, axis=(0,1))) * 255
        ts_s = (ts_s - np.min(ts_s, axis=(0,1))) / (np.max(ts_s, axis=(0,1)) - np.min(ts_s, axis=(0,1))) * 255
        ts_st = (ts_st - np.min(ts_st, axis=(0,1))) / (np.max(ts_st, axis=(0,1)) - np.min(ts_st, axis=(0,1))) * 255

        sk.imsave(result_path + "TS/" + folder + "/" + video_name + ".png", ts.astype(np.uint8), check_contrast=False)
        sk.imsave(result_path + "TS_t/" + folder + "/" + video_name + ".png", ts_t.astype(np.uint8), check_contrast=False)
        sk.imsave(result_path + "TS_s/" + folder + "/" + video_name + ".png", ts_s.astype(np.uint8), check_contrast=False)
        sk.imsave(result_path + "TS_st/" + folder + "/" + video_name + ".png", ts_st.astype(np.uint8), check_contrast=False)

        #plt.imshow(ts)
        #plt.show()
        #exit(-1)



if __name__ == '__main__':

   
    path = "E:/MoviesFights/"


    compute_TS(path_dataset = path + "Datasets/", result_path = "E:/MoviesFights/", n_clusters=8, miniBtach=True)
    

    