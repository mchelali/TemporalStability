# -*-coding: utf-8 -*-
import numpy as np
import skimage.io as sk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import classification_report, precision_score, cohen_kappa_score
from SatProc import tools
import pickle as pkl
from features import TemporelStability_faster, AttribueTemporelle
from sklearn.model_selection import LeaveOneOut
import evaluation
import time
from features.GLCM import getTexture_Parallel
import joblib
from features.TemporalStability_C import TemporelStability as TS

##################################################################
#                       Approche globale                         #
##################################################################

def getTemporalStability_Global(indice, disc_k, path_data, point, data='all', n_job=1, path_result=None):
    """
        #########################################################################
        #########################################################################
        #                                                                       #
        #                   Calcul de la stabilité temporelle                   #
        #                                                                       #
        #########################################################################
        #########################################################################
    """

    files_train = tools.readFiles(path_data, '.tif')
    files_train = sorted(files_train, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split(".")[0]))

    date_vec = tools.numJourAn_List(files_train)
    #date_vec = [tools.getRankday(f) for f in files_train]
    print ("Nbr Fichers ", len(files_train))

    X = []
    for file in files_train:
        img = sk.imread(file)
        if indice == "NDVI":
            ndvi = tools.toNDVI(img)
        elif indice == "NDWI":
            ndvi = tools.toNDWI(img)
        elif indice == "IB":
            ndvi = tools.toIB(img)
        elif indice == "exp_NDVI":
            ndvi = np.exp(tools.toNDVI(img))
        elif indice == "exp_NDWI":
            ndvi = np.exp(tools.toNDWI(img))
        elif indice == "exp_IB":
            ndvi = np.exp(tools.toIB(img)/(2**16 - 1))
        else:
            ndvi = np.mean(img, axis=2)

        X.append(ndvi)
    X = np.dstack(X)

    if data == 'all':
        st = TS.TemporelStability_Glouton(n_clusters=disc_k, miniBtach=False, data='all')#, n_job=n_job)
    elif data == '10':
        st = TS.TemporelStability_Glouton(n_clusters=disc_k, miniBtach=False, data='10')#, n_job=n_job)
    else:
        st = TS.TemporelStability_Glouton(n_clusters=disc_k, miniBtach=False, data='p')#, n_job=n_job)
        st.setROI(point)

    print("Quantification ")
    t0 = time.time()
    st.fit(date_vec, X)
    print("fin de la quantification; temps d'execution est ", time.time() - t0)

    return st

def getCorrelationOfEvolution_Global(path_data):
    """
        #########################################################################
        #########################################################################
        #                                                                       #
        #                       Calcul du type d'evolution                      #
        #                                                                       #
        #########################################################################
        #########################################################################
    """

    files_train = tools.readFiles(path_data, '.tif')
    files_train = sorted(files_train, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split(".")[0]))

    l, c, d = sk.imread(files_train[0]).shape  # Get the shape of image

    mask = 3
    attribue = AttribueTemporelle.AttribueTemporelle(files_train, masque=mask)

    t1 = time.time()
    print("START Calculate Cross Correlations AT", t1)
    attribue.fit_data()
    print("END AT", time.time() - t1)

    att = np.zeros((l, c))
    att[1:l - 1, 1:c - 1] = attribue.attribue.reshape((l - 2, c - 2))

def extract_feature_global(indice, disc_k, path_data, path_result):
    smt, nbt = getTemporalStability_Global(indice, disc_k, path_data)
    att = getCorrelationOfEvolution_Global(path_data)

    #########################################################################
    #########################################################################
    #                                                                       #
    #                       Enregistrer les images                          #
    #                                                                       #
    #########################################################################
    #########################################################################

    tools.saveImage(tools.normalisation2(nbt).astype(np.uint8), path_result + "nbt_" + indice + "_" + str(disc_k))
    tools.saveImage(tools.normalisation2(smt).astype(np.uint8), path_result + "smt_" + indice + "_" + str(disc_k))
    tools.saveImage(tools.normalisation2(att).astype(np.uint8), path_result + "att_ndvi_ndwi_ib_date_" + str(disc_k))

##################################################################
#                       Approche locale                          #
##################################################################

def getTemporelWindow(files_train):
    """

    :param files_train: liste des fichiers
    :return:
    """
    date_vec = tools.numJourAn_List(files_train)

    print (date_vec)
    window = []
    for i in range(1, 13):
        month = [file for file in files_train if int(re.findall("[0-9]{2}", file.split("/")[-1])[2]) == i]
        window.append(month)

    window2 = []
    for i in range(len(window) - 1):
        window2.append(window[i][len(window[i]) // 2:] + window[i + 1][0:len(window[i + 1]) // 2])

    all = []
    for i in range(len(window)):
        all.append(window[i])
        if i < 11:
            all.append(window2[i])
    return all

def getTemporalStability_local(indice, disc_k, path_data, point, data="all",  path_result=None):
    files_train = tools.readFiles(path_data, '.tif')
    files_train = sorted(files_train, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split(".")[0]))

    geoTransform, projection = tools.getGeoInformation(files_train[0])
    print ("Geo Transform : ", geoTransform)
    print ("Projection : ", projection)

    date_vec = tools.numJourAn_List(files_train)
    print ("Lecture des images")
    x_train = []

    for file in files_train:
        img = sk.imread(file)
        if indice == "NDVI":
            ndvi = tools.toNDVI(img)
        elif indice == "NDWI":
            ndvi = tools.toNDWI(img)
        elif indice == "IB":
            ndvi = tools.toIB(img)
        elif indice == "exp_NDVI":
            ndvi = np.exp(tools.toNDVI(img))
        elif indice == "exp_NDWI":
            ndvi = np.exp(tools.toNDWI(img))
        elif indice == "exp_IB":
            ndvi = np.exp(tools.toIB(img) / (2 ** 16 - 1))
        else:
            ndvi = np.mean(img, axis=2)

        x_train.append(ndvi)
    x_train = np.dstack(x_train)

    if data == 'all':
        st = TemporelStability.TemporelStability(n_clusters=disc_k, miniBtach=False, data='all')
    elif data == "10":
        st = TemporelStability.TemporelStability(n_clusters=disc_k, miniBtach=False, data='10')
    else:
        st = TemporelStability.TemporelStability(n_clusters=disc_k, miniBtach=False, data='p')
        st.setSelectedPoint(point)
    print ("Application du KMeans")
    st.fit(date_vec, x_train)
    # st.discritize(x_train)

    Temporal_window = getTemporelWindow(files_train)

    smt_all = []
    nbt_all = []
    startMax = []
    for i in range(len(Temporal_window)):
        date_vec_window = tools.numJourAn_List(Temporal_window[i])
        X = []
        for file in Temporal_window[i]:
            img = sk.imread(file)
            if indice == "NDVI":
                ndvi = tools.toNDVI(img)
            elif indice == "NDWI":
                ndvi = tools.toNDWI(img)
            elif indice == "IB":
                ndvi = tools.toIB(img)
            elif indice == "exp_NDVI":
                ndvi = np.exp(tools.toNDVI(img))
            elif indice == "exp_NDWI":
                ndvi = np.exp(tools.toNDWI(img))
            elif indice == "exp_IB":
                ndvi = np.exp(tools.toIB(img) / (2 ** 16 - 1))
            else:
                ndvi = np.mean(img, axis=2)

            X.append(ndvi)
        X = np.dstack(X)

        st.discritize(date_vec_window, X)

        print("Calcul de la stabilte temporelles | windows ", i + 1)

        smt = st.getSM_t()
        smt = smt.astype(np.float) / (max(date_vec_window) - min(date_vec_window))
        smt_all.append(smt)


        nbt = st.getNb_t()
        nbt = nbt.astype(np.float) / float(len(date_vec_window))
        nbt_all.append(nbt)

        # Enregistrement des images
        startMax_ = st.getMapStartMax()
        startMax_ = startMax_.astype(np.float) / (max(date_vec_window) - min(date_vec_window))
        startMax.append(startMax_)

        if not (path_result is None):
            tools.saveImage(img=tools.normalisation2(nbt[:, :, i]).astype(np.uint8),
                            path=path_result + indice + "/nbt_" + str(i) + indice + "_" + str(disc_k) + ".png",
                            geoTransform=geoTransform,
                            projection=projection)
            tools.saveImage(img=tools.normalisation2(smt[:, :, i]).astype(np.uint8),
                            path=path_result + indice + "/smt_" + str(i) + indice + "_" + str(disc_k) + ".png",
                            geoTransform=geoTransform,
                            projection=projection)
            tools.saveImage(img=tools.normalisation2(startMax[:, :, i]).astype(np.uint8),
                            path=path_result + indice + "/startMax_" + str(i) + indice + "_" + str(disc_k) + ".png",
                            geoTransform=geoTransform,
                            projection=projection)


    smt_all = np.dstack(smt_all)
    nbt_all = np.dstack(nbt_all)
    startMax = np.dstack(startMax)

    if not (path_result is None):
        np.save(path_result + indice + "/disc" + indice + "_" + str(disc_k) + "_smt_All", smt_all)
        np.save(path_result + indice + "/disc" + indice + "_" + str(disc_k) + "_nbt_All", nbt_all)
        np.save(path_result + indice + "/disc" + indice + "_" + str(disc_k) + "_startMax_All", startMax)

    return st, smt_all, nbt_all, startMax

def getCorrelationOfEvolution_local(path_data):
    files_train = tools.readFiles(path_data, '.tif')
    files_train = sorted(files_train, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split(".")[0]))

    l, c, d = sk.imread(files_train[0]).shape  # Get the shape of image

    Temporal_window = getTemporelWindow(files_train)
    att_all = []

    for i in range(len(Temporal_window)):
        mask = 3
        attribue = AttribueTemporelle.AttribueTemporelle(Temporal_window[i], masque=mask)

        t1 = time.time()
        print("START Calculate Cross Correlations AT", t1)
        attribue.fit_data()
        print("END AT", time.time() - t1)

        att = np.zeros((l, c))
        att[1:l - 1, 1:c - 1] = attribue.attribue.reshape((l - 2, c - 2))

        att = tools.normalisation2(att).astype(np.uint8)
        att_all.append(att)

    att_all = np.dstack(att_all)
    return att_all

def extract_feature_local(indice, disc_k, path_data, path_result):
    if not os.path.exists(path_result + "/" + indice):
        os.makedirs(path_result + "/" + indice)

    path_result1 = path_result + "/" + indice + "/"

    smt, nbt = getTemporalStability_local(indice, disc_k, path_data)
    att = getCorrelationOfEvolution_local(path_data)

    for i in range(att.shape[2]):
        tools.saveImage(tools.normalisation2(nbt[:, :, i]).astype(np.uint8),
                        path_result1 + "nbt_" + str(i) + indice + "_" + str(disc_k))
        tools.saveImage(tools.normalisation2(smt[:, :, i]).astype(np.uint8),
                        path_result1 + "smt_" + str(i) + indice + "_" + str(disc_k))
        tools.saveImage(tools.normalisation2(att[:, :, i]).astype(np.uint8),
                        path_result + "att_" + str(i) + "ndvi_ndwi_ib_date_" + str(disc_k))

    np.save(path_result1 + "/" + indice + "/disc" + indice + "_" + str(disc_k) + "smt_All_" + indice, smt)
    np.save(path_result1 + "/" + indice + "/disc" + indice + "_" + str(disc_k) + "nbt_All_" + indice, nbt)
    np.save(path_result + "/att_ndv_ndwi_ib_date", att)

###############################################################
###             Extraction de texture                       ###
###############################################################

def extractTextureHaralick(path_data, path_result):
    files = tools.readFiles(path_data, ".tif")

    for file in files:
        filename = file.split('/')[-1].split(".")[0]
        # Lecture des donnees
        img = sk.imread(file)

        gray = (img[:, :, 1] + img[:, :, 2] + img[:, :, 3]) / 3
        gray = tools.normalisation2(gray).astype(np.uint8)

        w = 3

        direction = ["0", "PIdiv2", "PI", "3PIdiv2"]
        angle = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        t = time.time()
        contrast, dissimilarity, homogeneity, energy, correlation, ASM = getTexture_Parallel(gray, 3, [1], angle, 5)
        print("end texture extraction after ", (time.time()-t)/60.)

        np.save(path_result + filename + "_contrast", contrast)
        np.save(path_result + filename + "_dissimilarity", dissimilarity)
        np.save(path_result + filename + "_homogeneity", homogeneity)
        np.save(path_result + filename + "_energy", energy)
        np.save(path_result + filename + "_correlation", correlation)
        np.save(path_result + filename + "_ASM", ASM)

        # Normalisation
        for i in range(contrast.shape[2]):
            contrastraster = tools.normalisation2(contrast[:, :, i])
            contrastraster = contrastraster.astype(int)

            dissimilarityraster = tools.normalisation2(dissimilarity[:, :, i])
            dissimilarityraster = dissimilarityraster.astype(int)

            homogeneityraster = tools.normalisation2(homogeneity[:, :, i])
            homogeneityraster = homogeneityraster.astype(int)

            energyraster = tools.normalisation2(energy[:, :, i])
            energyraster = energyraster.astype(int)

            correlationraster = tools.normalisation2(correlation[:, :, i])
            correlationraster = correlationraster.astype(int)

            ASMraster = tools.normalisation2(ASM[:, :, i])
            ASMraster = ASMraster.astype(int)

            tools.saveImage(contrastraster,
                            path_result+filename+"_contsrat_" + str(w) + "angle" + direction[i] + ".png")
            tools.saveImage(dissimilarityraster,
                            path_result+filename+"_dissimilarity_" + str(w) + "angle" + direction[i] + ".png")
            tools.saveImage(homogeneityraster,
                            path_result+filename+"_homogeneity_" + str(w) + "angle" + direction[i] + ".png")
            tools.saveImage(energyraster,
                            path_result+filename+"_energy_" + str(w) + "angle" + direction[i] + ".png")
            tools.saveImage(correlationraster,
                            path_result+filename+"_correlation_" + str(w) + "angle" + direction[i] + ".png")
            tools.saveImage(ASMraster,
                            path_result+filename+"_asm_" + str(w) + "angle" + direction[i] + ".png")

###############################################################