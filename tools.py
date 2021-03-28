#-*-coding: utf-8 -*-
import numpy as np
import skimage.io as sk
import TemporelStability as TS
import matplotlib.pyplot as plt
import cv2
import os
import gdal

def getFiles(path, extension):
    """
        Cette fonction permet d'identifier les fichier dans un dossier et de retourner ceux qui ont certaine extension
    :param path: chemin du dossier
    :return: files: liste des images .tif en chemin absolue
    """
    
    files = []
    for (dirpath, dirname, filesname) in os.walk(path):
        for filename in filesname:
            if filename.endswith(extension):
                #print(os.path.join(dirpath, filename))
                files.append(os.path.join(dirpath, filename))
                
    return files

def readSTIS(path_data, extension, index="MS"):
    files_train = getFiles(path_data, extension)
    files_train = sorted(files_train, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split(".")[0].split("-")[0]))

    #date_vec = numJourAn_List(files_train)
    print ("Nbr Fichers ", len(files_train))

    length = len(files_train)
    height, width, channels  = sk.imread(files_train[0]).shape

    if index=="MS":
        frames = np.zeros((length, height, width, channels))
    else:
        frames = np.zeros((length, height, width))

    for idx, file in enumerate(files_train) :
        img = sk.imread(file)
        
        if index == "NDVI":
            img = toNDVI(img)
        elif index == "NDWI":
            img = toNDWI(img)
        elif index == "IB":
            img = toIB(img)
        frames[idx] = img

        

    return frames

def readVideo(path, color=True):
    """
        path: path to the video
        color: if True, read frames in color mode else read them in gray mode
    """
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3 if color else 1 #int(cap.get(cv2.CAP_PROP_CHANNEL))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    #print(length, channels, height, width, fps)
    if color:
        frames = np.zeros((length, height, width, channels), dtype=np.uint8)
    else:
        frames = np.zeros((length, height, width), dtype=np.uint8)
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video stream or file")

    # Read until video is completed
    i = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(frame.shape)
            frames[i] = frame
            i+=1
        # Break the loop
        else: 
            break
    # When everything done, release the video capture object
    cap.release()
    
    return frames

def toNDVI(a):
    """
        On change d'espace de couleur de IR.RGB vers NDVI qui ppermet de mieux localiser la vegetation
    """
    #ok = (a[:,:,0].astype(np.double)+a[:,:,1].astype(np.double)) != 0
    ndvi = ((a[:, :, 0].astype(np.double)-a[:, :, 1].astype(np.double)) / (a[:, :, 0].astype(np.double)+a[:, :, 1].astype(np.double)))
    ndvi[np.isnan(ndvi)] = -1
    ndvi[ndvi == np.inf] = 1.0
    ndvi[ndvi == -float(np.inf)] = -1.0
    return ndvi

def toNDWI(a):
    """
        On change d'espace de couleur de IR.RGB vers NDWI qui ppermet de mieux localiser l'eaux
    """
    ndwi = ((a[:, :, 2].astype(np.double) - a[:, :, 0].astype(np.double)) / (a[:, :, 2].astype(np.double) + a[:, :, 0].astype(np.double)))
    ndwi[np.isnan(ndwi)] = -1
    ndwi[ndwi == np.inf] = 1.0
    ndwi[ndwi == -float(np.inf)] = -1.0
    return ndwi

def toIB(a):
    """
        Calcul de l'indice de briance
    """
    return (np.sqrt(a[:, :, 1].astype(np.double)**2 + a[:, :, 0].astype(np.double)**2))

def cropSTIS(pathFiles, pathToSave, x1, x2, y1, y2, geoInf=True):
    """

    :param pathFiles: chemin contenant les image à croppé
    :param pathToSave: Chemin ou enregistrer les images croppé
    :param x1: x du point de depart
    :param x2: x du point d'arriver
    :param y1: y du point de depart
    :param y2: y du point d'arriver
    :param geoInf: Si True alors on garde le geo referencement sinon on le perd
    :return: None
    """

    files = getFiles(pathFiles, ".tif")
    files = sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split(".")[0].split("-")[0]))

    if geoInf:
        # recuperer l'ancien georeferencement
        referenced_image = gdal.Open(files[0])
        transform = referenced_image.GetGeoTransform()
        new_transformation = (transform[0] + y1 * transform[1], transform[1], transform[2], transform[3] + x1 * transform[5], transform[4],  transform[5])

        projection = referenced_image.GetProjection()
        print("Olt transformation \t", transform)
        print("New Transformation \t", new_transformation)

        print(projection)
    else:
        projection = None
        new_transformation = None


    for i in range(len(files)):
        img = sk.imread(files[i])[x1:x2, y1:y2, :]
        print(img.shape)
        filename = os.path.splitext(os.path.basename(files[i]))[0].split(".")[0] + ".tif"
        #filename = datenumjouran(date_vec[i], 2017)
        #filename = str(filename[2]) + ("%02d" % (filename[1])) + ("%02d" % (filename[0])) + ".tif"
        print ("---------> ",  pathToSave+filename)
        saveImage(img=img,
                  path=pathToSave+filename,
                  geoTransform=new_transformation,
                  projection=projection,
                  type=gdal.GDT_UInt16)

def saveImage(img, path, geoTransform=None, projection=None, type=gdal.GDT_Byte):
    if len(img.shape) == 2:
        rows, cols = img.shape
        d = 1
    else:
        rows, cols, d = img.shape

    #print(rows, cols)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path,  cols, rows, d, type)#gdal.GDT_Byte)
    if outdata is None:
        return ValueError( path)

    if not geoTransform is None:
        outdata.SetGeoTransform(geoTransform)  ##sets same geotransform as input
    if not projection is None:
        outdata.SetProjection(projection)  ##sets same projection as input

    if d == 1:
        outdata.GetRasterBand(1).WriteArray(img)
    else:
        for i in range(d):
            outdata.GetRasterBand(i+1).WriteArray(img[:,:, i])

    outdata.FlushCache()  ##saves to disk!!
    outdata = None

