# -*-coding: utf-8 -*-
import matplotlib
# matplotlib.use("Qt4Agg")

import numpy as np
import skimage.io as sk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import os
import pickle as pkl
import re
import random
import TemporalStability as c_func

class TemporelStability:
    def __init__(self, n_clusters=2, miniBtach=False, data="all"):
        """

        """
        self.selectedPoint = None
        # Initialisation du KMeans
        if miniBtach == True:
            self.km = MiniBatchKMeans(
                n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=100, random_state=0)
        else:
            self.km = KMeans(n_clusters=n_clusters,
                             init='k-means++', max_iter=100, random_state=0)

        self.data = data

        self.globale = False
        self.locale = False

    def setSelectedPoint(self, XTrue):
        """
            :param trueX: image of the ROIs (1 if consideried pixels)
            :return:
        """
        self.selectedPoint = XTrue

    def fit(self, X):
        """
            :param *X: list of STIS
            :return:
        """
        if len(X) == 0:
            print("Give at least 1 STIS('ndarray' where shape is nuber of line and nuber of cols and dim is the number of images where each image is token at time t)")
            return

        if X.ndim == 4:
            self.l, self.c, self.d, self.t = X.shape
        else:
            self.l, self.c, self.t = X.shape
            self.d = 1

        #print(self.l, self.c, self.d, self.t)
        if self.data == "10":
            self.nbPix = int(self.l * self.c * self.t * 0.1)
            #print("nb pixel: ", self.nbPix)
            idx = np.random.randint(self.l * self.c * self.t, size=self.nbPix)
            #print("shape randint ", idx.shape)
            self.features = np.random.choice(X.reshape((-1, self.d)), self.nbPix)
        elif self.data != 'all':
            if self.selectedPoint is None:
                print("Please give an array where intersted point are equal to 1")
                return -1
            else:
                #print("Discrize of selected points ")
                self.features = X[self.selectedPoint != 0].reshape(
                    (-1, 1))
        else:
            #print("discrize all points ")
            self.features = X.reshape((-1, self.d))

        #print("X shape ", X.shape)
        #print("features shape ",  self.features.shape)
        #exit(0)
        #  passer les donnees au kmeans pour trouver les centroides
        self.km.fit(self.features)
        #print(self.km.cluster_centers_)
        self.cluster = (self.km.cluster_centers_.squeeze())

        X = X.reshape((-1, self.d))
        self.serie = self.km.predict(X).reshape((self.l, self.c, self.t))

        """serie_tmp = []
        for i in range(self.t):
            s = self.km.predict(X[:, :, i].reshape((-1, self.d)))
            #s = np.choose(s, self.cluster)
            #aa_ = tools.normalisation2(s.reshape((self.l, self.c))).astype(np.uint8)
            #aa_[self.selectedPoint == 0 ] = 0
            #tools.saveImage_(aa_, "E:/newSITS/vergers/img_k4/"+str(i)+".tif")
            serie_tmp.append(s.reshape((self.l, self.c)))
            #plt.imshow(s.reshape((self.l, self.c)))
            #plt.pause(0.1)
        #plt.show()

        serie_tmp = np.ascontiguousarray(np.dstack(serie_tmp), dtype=np.double)
        self.serie = serie_tmp"""

    def getDiskrizedSits(self):
        return self.serie[0]

    def setROI(self, XTrue):
        """
                Set the image of the ROI. interested pixels are those who are different from 0
            :param XTrue: Image of the ROI
            :return:
        """
        self.selectedPoint = XTrue

    def discritize(self, date_vec, *X):
        """
            This funtion is used just to discrize a new STIS
            ====> it is developed to be used for the local method
        :param *X: list of STIS
        :return:
        """
        if len(X) == 0:
            print("Give at least 1 STIS('ndarray' where shape is nuber of line and nuber of cols and dim is the number of images where each image is token at time t)")
            return -1

        self.l, self.c, self.d = X[0].shape
        self.serie = []
        for stis in X:
            self.serie_tmp = []
            for i in range(self.d):
                s = self.km.predict(stis[:, :, i].reshape((-1, 1)))
                s = np.choose(s, self.km.cluster_centers_.squeeze())
                self.serie_tmp.append(s.ravel())

            self.serie.append(np.array(self.serie_tmp).T)

        # self.__fit__(date_vec)

    def get_TS(self, date_vec):
        date_vec = np.ascontiguousarray(np.array(date_vec), dtype=np.int)
        #print("Python print\n", date_vec)
        # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
        self.ts = np.ascontiguousarray(np.zeros((self.l, self.c, 3)), dtype=np.double)
        c_func.getTemporalStability(self.serie.astype(np.double), date_vec, self.ts)

        self.ts[:, :, 0] = self.ts[:, :, 0] / (max(date_vec) - min(date_vec))
        self.ts[:, :, 1] = self.ts[:, :, 1] / len(date_vec)
        self.ts[:, :, 2] = self.ts[:, :, 2] / (max(date_vec) - min(date_vec))

        if not (self.selectedPoint is None):
            self.ts[self.selectedPoint == 0] = 0
        
        return self.ts

    def get_Reconstracted_TS(self, date_vec):
        date_vec = np.ascontiguousarray(np.array(date_vec), dtype=np.int)
        print("Python print\n", date_vec)
        # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
        ts = np.ascontiguousarray(np.zeros((self.l, self.c, self.d)), dtype=np.double)
        ts = c_func.getReconstruncted_TemporalStability(self.serie[0].astype(np.double), date_vec, ts)
        # print(ts.shape)
        self.reco_series= ts

        if not (self.selectedPoint is None):
            self.reco_series[self.selectedPoint == 0] = 0

        return self.reco_series


    def getMS(self):
        return self.ts[:, :, 0]

    def getNb(self):
        return self.ts[:, :, 1]

    def getMSS(self):
        return self.ts[:, :, 2]

    def get_TS_local(self, date_vec, window):

        self.smt_local = list()
        self.nbt_local = list()
        self.map_stratMax_local = list()
        for i in range(0, len(date_vec), window):
            print("========= ", i, " =========")
            print(date_vec[i:i + window])

            date_vec_ = np.ascontiguousarray(
                np.array(date_vec[i:i + window]), dtype=np.int)
            # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
            ts = np.ascontiguousarray(
                np.zeros((self.l, self.c, 3)), dtype=np.double)
            ts = c_func.getTemporalStability(
                self.serie[0][:,:,i:i + window].astype(np.double), date_vec_, ts)
            # print(ts.shape)
            if not (self.selectedPoint is None):
                ts[self.selectedPoint == 0] = 0

            self.smt_local.append(ts[:, :, 0])
            self.nbt_local.append(ts[:, :, 1])
            self.map_stratMax_local.append(ts[:, :, 2])
            del ts

        self.smt_local = np.dstack(self.smt_local)
        self.nbt_local = np.dstack(self.nbt_local)
        self.map_stratMax_local = np.dstack(self.map_stratMax_local)

    def getSM_t_local(self):
        return self.smt_local

    def getNb_t_local(self):
        return self.nbt_local

    def getMapStartMax_local(self):
        return self.map_stratMax_local

    ############## Relaxation Temporelle ##################
    def get_TS_t(self, date_vec):
        date_vec = np.ascontiguousarray(np.array(date_vec), dtype=np.int)
        self.ts_t = np.ascontiguousarray(np.zeros((self.l, self.c, 3)), dtype=np.double)
        c_func.getTemporalStability_temp(self.serie.astype(np.double), date_vec, self.ts_t)

        self.ts_t[:, :, 0] = self.ts_t[:, :, 0] / (max(date_vec) - min(date_vec))
        self.ts_t[:, :, 1] = self.ts_t[:, :, 1] / len(date_vec)
        self.ts_t[:, :, 2] = self.ts_t[:, :, 2] / (max(date_vec) - min(date_vec))

        if not (self.selectedPoint is None):
            self.ts_t[self.selectedPoint == 0] = 0
        
        return self.ts_t

    def getMS_t(self):
        return self.ts_t[:, :, 0]

    def getNb_t(self):
        return self.ts_t[:, :, 1]

    def getMSS_t(self):
        return self.ts_t[:, :, 2]

    def get_Reconstracted_TS_temp(self, date_vec):
        date_vec = np.ascontiguousarray(np.array(date_vec), dtype=np.int)
        print("Python print\n", date_vec)
        # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
        ts = np.ascontiguousarray(np.zeros((self.l, self.c, self.d)), dtype=np.double)
        ts = c_func.getReconstruncted_TemporalStability_temp(self.serie[0].astype(np.double), date_vec, ts)
        # print(ts.shape)
        self.reco_series_temp= ts

        if not (self.selectedPoint is None):
            self.reco_series_temp[self.selectedPoint == 0] = 0

        return self.reco_series_temp

    ############## Relaxation Spatiale ##################
    def get_TS_s(self, date_vec):
        """
            Calcule de la stabilite temporelle dans un voisinnage (tyuau large ==> tunel)
            :param date_vec:
            :return:
        """
        date_vec = np.ascontiguousarray(
            np.array(date_vec), dtype=np.int)
        # print(date_vec.shape)
        # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
        self.ts_s = np.ascontiguousarray(np.zeros((self.l, self.c, 3)), dtype=np.double)
        c_func.getTemporalStability_spatio(self.serie.astype(np.double), date_vec, self.ts_s )

        self.ts_s[:, :, 0] = self.ts_s[:, :, 0] / (max(date_vec) - min(date_vec))
        self.ts_s[:, :, 1] = self.ts_s[:, :, 1] / len(date_vec)
        self.ts_s[:, :, 2] = self.ts_s[:, :, 2] / (max(date_vec) - min(date_vec))

        if not (self.selectedPoint is None):
            self.ts_s[self.selectedPoint == 0] = 0
        
        return self.ts_s

    def getMS_s(self):
        return self.ts_s[:, :, 0]

    def getNb_s(self):
        return self.ts_s[:, :, 1]

    def getMSS_s(self):
        return self.ts_s[:, :, 2]

    def get_Reconstracted_TS_spatio(self, date_vec):
        date_vec = np.ascontiguousarray(np.array(date_vec), dtype=np.int)
        print("Python print\n", date_vec)
        # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
        ts = np.ascontiguousarray(np.zeros((self.l, self.c, self.d)), dtype=np.double)
        ts = c_func.getReconstruncted_TemporalStability_spatio(self.serie[0].astype(np.double), date_vec, ts)
        # print(ts.shape)
        self.reco_series_spatio= ts

        if not (self.selectedPoint is None):
            self.reco_series_spatio[self.selectedPoint == 0] = 0

        return self.reco_series_spatio

    ############## Relaxation Spatio-temporelle ##################

    def get_TS_st(self, date_vec):
        """
            Calcule de la stabilite temporelle dans un voisinnage (tyuau large ==> tunel) et permettre de sauter une image
            si different pour voir ce qui vient apres
            :exemple:
               v= [[2 2 2 1 6 5]
                  [3 3 1 1 4 4]
                  [1 1 1 1 2 6]]
                encode_spatioTempShift(v)
                output:
                    [5, 1]

            :param date_vec:
            :return:
        """
        date_vec = np.ascontiguousarray( np.array(date_vec), dtype=np.int)
        self.ts_st = np.ascontiguousarray(np.zeros((self.l, self.c, 3)), dtype=np.double)
        c_func.getTemporalStability_spatiotemp(self.serie.astype(np.double), date_vec, self.ts_st )

        self.ts_st[:, :, 0] = self.ts_st[:, :, 0] / (max(date_vec) - min(date_vec))
        self.ts_st[:, :, 1] = self.ts_st[:, :, 1] / len(date_vec)
        self.ts_st[:, :, 2] = self.ts_st[:, :, 2] / (max(date_vec) - min(date_vec))

        if not (self.selectedPoint is None):
            self.ts_st[self.selectedPoint == 0] = 0

        return self.ts_st

    def getMS_st(self):
        return self.ts_st[:, :, 0]

    def getNb_st(self):
        return self.ts_st[:, :, 1]

    def getMSS_st(self):
        return self.ts_st[:, :, 2]

    def get_Reconstracted_TS_spatiotemp(self, date_vec):
        date_vec = np.ascontiguousarray(np.array(date_vec), dtype=np.int)
        print("Python print\n", date_vec)
        # print("Python: Calcul des MS, NB et MSS;\n dates = ", date_vec)
        ts = np.ascontiguousarray(np.zeros((self.l, self.c, self.d)), dtype=np.double)
        ts = c_func.getReconstruncted_TemporalStability_spatiotemp(self.serie[0].astype(np.double), date_vec, ts)
        # print(ts.shape)
        self.reco_series_spatiotemp= ts

        if not (self.selectedPoint is None):
            self.reco_series_spatiotemp[self.selectedPoint == 0] = 0

        return self.reco_series_spatiotemp
