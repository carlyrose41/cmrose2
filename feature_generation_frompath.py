import pandas as pd
import sys
import csv
import logging
import numpy as np
import logging
import pickle
import scipy.ndimage as ndi
import scipy.stats as stats
import multiprocessing as mp
from functools import partial
from time import gmtime, strftime, localtime
import SimpleITK as sitk
sys.path.append('/SwansonLab')
sys.path.append('/SwansonLab/PNTiLab')
import DjangoORM
# from PNTiLab import secret_settings
from Image import Image, ImageSubtype
from Measurement import Measurement
from Patient import Patient
import image_preprocessing
import os

mainDirectory = "/Users/m209884/Dropbox/Imaging_SexDifferences/iMacFolders/tomas/"
dataDirectory = "/Users/m209884/Dropbox/Imaging_SexDifferences/Analysis/data"

# mainDirectory = "/Users/kwsingleton/Dropbox/Imaging_SexDifferences/iMacFolders/tomas/"
# dataDirectory = "/Users/kwsingleton/Dropbox/Imaging_SexDifferences/Analysis/data"

# mainDirectory = "/Users/andreahawkins-daarud/Dropbox/Imaging_SexDifferences/iMacFolders/tomas/"
# dataDirectory = "/Users/andreahawkins-daarud/Dropbox/Imaging_SexDifferences/Analysis/data"

# caseFilePath = 'images_1.csv' # Carly's females, has stripped dates
# caseFilePath = 'newFemales.csv'
# caseFilePath = 'NewMales_FeatureExtract_Format_82019_MinusProblems.csv'
# caseFilePath = 'New_males_test.csv'
caseFilePath = 'NewMales_August2019.csv'
# caseFilePath = 'NewFemales_Sept2019.csv'

# logFilePath = 'logs/feature_generation.log'
logFilePath = 'logs/feature_generation_newMales.log'
# logFilePath = 'logs/feature_generation_newFemales.log'

# featureFileName = 'feature_data.csv'
# tabularFileName = 'tabular_data.csv'

# featureFileName = 'feature_data_newMales.csv'
# tabularFileName = 'tabular_data_newMales.csv'
featureFileName = 'feature_data_newMales_August2019.csv'
tabularFileName = 'tabular_data_newMales_August2019.csv'

featureFileName = 'feature_data_newMales_August2019_kws.csv'
tabularFileName = 'tabular_data_newMales_August2019_kws.csv'


# featureFileName = 'feature_data_newFemales.csv'
# tabularFileName = 'tabular_data_newFemales.csv'
# featureFileName = 'feature_data_newFemales_Sept2019.csv'
# tabularFileName = 'tabular_data_newFemales_Sept2019.csv'

# roiFileName = 'tabular_data_newMales_roi.csv'
# roiFileName = 'tabular_data_newFemales_roi.csv'
roiFileName = 'tabular_data_newMales_roi_August2019.csv'
roiFileName = 'tabular_data_newMales_roi_August2019_kws.csv'

# roiFileName = 'tabular_data_newFemales_roi_Sept2019.csv'
# roiFileName = 'tabular_data_newCases_roi_Sept2019'

# replacementPath = '/Users/kwsingleton/Dropbox/Imaging_SexDifferences/ImagingSexDifferences_NewFemales'
replacementPath = '/Volumes/MLDL_Projects/ImagingSexDifferences/NewMales_August2019'
# replacementPath = '/Volumes/MLDL_Projects/ImagingSexDifferences/NewFemales_Sept2019'

def loadCases():
    '''
    Collect ids for each date for analysis. Ids include both registered ids and the base id for that date
    Return in dict {} format with the date as the key and value as a list of the ids
    '''

    df = pd.read_csv(os.path.join(dataDirectory, caseFilePath))

    patients = df['patient'].unique()
    cases = []

    for patient in patients:
        for date in list(df.loc[df['patient'] == patient]['date_of_image'].unique()):
            data = df.loc[(df['date_of_image'] == date) & (df['patient'] == patient)]
            # 'date_of_image'.format(datetime.datetime.strptime(date, '%m/%d/%YT%H-%M-%S+00-00').strftime('%Y-%m-%dT%H-%M-%S+00-00'))
            # date = date.format(strftime('%Y-%m-%dT%H-%M-%S+00-00'))
            ids = []
            for index, row in data.iterrows():
                baseId = int(row['base_id'])
                ids.append(int(row['registered_id']))
            if len(ids) >= 1:
                ids.append(baseId)
                info = {}
                info['patient'] = patient
                info['date'] = date
                info['ids'] = ids
                info['base_id'] = baseId
                cases.append(info)
    return cases

# image.image_file_path = '/PatientImages/{}/{}/{}_Normalized_{}.nii.gz'.format(image.patient.patient_number, image.image_date.strftime('%Y-%m-%dT%H-%M-%S+00-00'), image.image_subtype.name, image.id)

# im.image_date.strftime('%Y-%m-%dT%H-%M-%S+00-00'    

# for case in cases:
#     ids = case['ids']
#     images = loadImages(ids, date=case['date'])

def getROICoords(cases):

    coordinateInfo = {}
    sub_region = {}
    all_data = pd.DataFrame(data=None, columns=['case', 'coords', 'sub_region'])
    GRID_SPACING = 8
    for case in cases:
        caseName = case['patient'] + '_' + case['date']

        logging.info('Starting case {}'.format(caseName))

        images = loadImages(case['ids'], date=case['date']) 
        t1, t1gd, t2, flair = image_preprocessing.getIndividualModalities(images)

        roi_matrix = None
        # t1_roiMatrix = np.zeros(images[0].matrix.shape)
        # t1gd_roiMatrix = np.zeros(images[0].matrix.shape)
        # t2_roiMatrix = np.zeros(images[0].matrix.shape)
        # flair_roiMatrix = np.zeros(images[0].matrix.shape)

        for image in images:
            print(image.id)
            m = list(Measurement.objects.filter(image_id=image.id, status='A', measurement_type_id=1))
            
            if m is not None and len(m) > 0:
                m = m[0]
            else:
                print('===> ERROR: No measurement was found in database! Skipping...')
                continue
            try:
                m.file_path = image.image_file_path.replace('Normalized', 'ROI')
                m.loadROI(image)
                print("ROI shape {}".format(m.roi_matrix.shape))
                print("Image shape {} {}".format(image.matrix.shape, image.sitk_image.GetSize()))
                if m.roi_matrix.shape != image.matrix.shape:
                    print('===> WARNING: Possible matrix size mismatch! ROI Size {}, Image Size {}'.format(m.roi_matrix.shape, image.matrix.shape))
            except Exception as e:
                print(e)
                print("===> ERROR: Could not load ROI. Skipping...")
                continue

            #roi_matrix = np.zeros(images[0].matrix.shape)
            print('{} imageshape {} measurement'.format(images[0].matrix.shape, m.roi_matrix.shape))

            # If first pass, initialize storage matrix
            # Use roi_matrix instead of image.matrix due to common shape mismatches
            if roi_matrix is None:
                roi_matrix = np.zeros(m.roi_matrix.shape)

            m.roi_matrix = np.floor(m.roi_matrix)

            currentSubtype = ImageSubtype.get_modality(image.image_subtype.id)
            if  currentSubtype == 'T1':
                roi_matrix = roi_matrix + m.roi_matrix
            if currentSubtype == 'T1Gd':
                roi_matrix = roi_matrix + 2*m.roi_matrix
            if currentSubtype == 'T2':
                roi_matrix = roi_matrix + 4*m.roi_matrix            
            if currentSubtype == 'FLAIR':
                roi_matrix = roi_matrix + 8*m.roi_matrix 

            print('>>> {} MATRIX SUM: {}'.format(currentSubtype, np.sum(roi_matrix)))
        
        print('>>> TOTAL MATRIX SUM: {}'.format(np.sum(roi_matrix)))
        # case_data = getCoordinates(GRID_SPACING, roi_matrix, caseName)

        if roi_matrix is None:
            print('===> ERROR: Zero (0) ROI files could be loaded! Considering removing case...')
            coordinateInfo[caseName] = []
        else:
            coordinateInfo[caseName] = getCoordinates(GRID_SPACING, roi_matrix)

        fullCoordinateInfo = {**coordinateInfo}
        
        # case_data = getCoordinates(GRID_SPACING, roiMatrix, caseName) 
        # all_data.append(case_data)

    # return all_data
    return fullCoordinateInfo


#Defined to replace the last underscore in file path
def replace_right(source, target, replacement, replacements=None):
    return replacement.join(source.rsplit(target, replacements))

def loadImages(ids, date=''):
    '''
    Given data frame with image_id and date_of_image colum,
    load image objects for that date
    '''
    
    # images = list(map(lambda x: Image(id=x, productionDatabase=True), ids))
    images = list(map(lambda x: Image.objects.get(id=x), ids))
    loadedImages = []
    for image in images:
        try: 
            # image.load_from_dicom_stack()
            if not '.nii' in image.image_file_path:
                image.image_file_path = '/PatientImages/{}/{}/{}_Normalized_{}.nii.gz'.format(image.patient.patient_number, image.image_date.strftime('%Y-%m-%dT%H-%M-%S+00-00'), image.image_subtype.name, image.id)

            # print('>>> old path: {}'.format(image.image_file_path))
            image.image_file_path = image.image_file_path.replace('/PatientImages', replacementPath)

            if image.image_file_path.find('T1GD'):
                logging.info("Converting UPPERCASE T1GD to lowercase T1Gd")
                image.image_file_path = image.image_file_path.replace('T1GD', 'T1Gd')

            # print('>>> new path: {}'.format(image.image_file_path))
            if image.image_file_path.find('Registered'):
                image.image_file_path = image.image_file_path.replace('Registered', 'Normalized')
            else:
                image.image_file_path = replace_right(image.image_file_path, "_", "_Normalized_", 1)
            
            # print('>>> new new path: {}'.format(image.image_file_path))

            image.loadFromDicomStack()
            loadedImages.append(image)
            print("Succeeded loading {}".format(image.id))
            logging.info("Succeeded loading {}".format(image.id))
        except Exception as e:
            print("Error loading {}".format(image.id))
            logging.error('{} did not load properly'.format(image.id))
            logging.error('{}'.format(e))

    return loadedImages

def getCoordinates(GRID_SPACING, image, caseName = None):
    coords = []
    sub_region = []
    case = []
    slices = image.shape[2]
    for index in range(slices):
        roiSlice = image[:,:, index]
        for x in range(roiSlice.shape[0]): 
            if x % GRID_SPACING != 0:
                continue
            for y in range(roiSlice.shape[1]):
                if y % GRID_SPACING == 0 and roiSlice[x,y]:
                    coords.append((x,y,index))
                    # case.append(caseName)
                    # if roiSlice[x,y] == 1:
                    #     sub_region.append((1,0,0,0))
                    # elif roiSlice[x,y] == 2:
                    #     sub_region.append((0,1,0,0))
                    # elif roiSlice[x,y] == 4:
                    #     sub_region.append((0,0,1,0))
                    # elif roiSlice[x,y] == 8:
                    #     sub_region.append((0,0,0,1))
                    # elif roiSlice[x,y] == 3:
                    #     sub_region.append((1,1,0,0))
                    # elif roiSlice[x,y] == 5:
                    #     sub_region.append((1,0,1,0))
                    # elif roiSlice[x,y] == 9:
                    #     sub_region.append((1,0,0,1))
                    # elif roiSlice[x,y] == 7:
                    #     sub_region.append((1,1,1,0))
                    # elif roiSlice[x,y] == 11:
                    #     sub_region.append((1,1,0,1))
                    # elif roiSlice[x,y] == 13:
                    #     sub_region.append((1,0,1,1))
                    # elif roiSlice[x,y] == 14:
                    #     sub_region.append((0,1,1,1))
                    # elif roiSlice[x,y] == 15:
                    #     sub_region.append((1,1,1,1))
                    # elif roiSlice[x,y] == 6:
                    #     sub_region.append((0,1,1,0))
                    # elif roiSlice[x,y] == 10:
                    #     sub_region.append((0,1,0,1))
                    # elif roiSlice[x,y] == 12:
                    #     sub_region.append((0,0,1,1))
                
    # data = {'case':case, 'coordinates':coords, 'sub_region':sub_region}
    # df = pd.DataFrame(data)

    # return df

    return coords


def tumorRegion():
    cases = loadCases()
    # all_data = getROICoords(cases)
    coordinateInfo = getROICoords(cases)

    cases = []
    coords = []
    for key in coordinateInfo:
        cases += [key] * len(coordinateInfo[key])
        coords += coordinateInfo[key]

    df = pd.DataFrame({'case' : cases, 'coordinates' : coords, 'ROI' : [True] * len(cases)}, columns = ['case', 'coordinates', 'ROI'])
    df['coordinates'] = df['coordinates'].apply(lambda x: str(x).replace('(', '').replace(')', '').replace(' ', ''))
    print(df.head())

    all_data = df

    # save so we dont have to reload
    # with open(os.path.join(dataDirectory, 'roi_coords.pickle'), 'wb') as f:
    #     pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

    return all_data

def extractRect(image, c):
    """
    coordinate 'c' contains two array indicies, in 2D or 3D
        
    Each index is a tuple corresponding to the (z,y,x) array axes, ie
    the first value in the tuple indicates which image (slice) in the volume
    to use. The second value in the tuple indicates which row in the volume,
    and the last value in the tuple indicates which column in the volume.
        
    """
    if image.size == 0:
        print("Image Empty")

    # print(len(c))
    # print(c)


    subImage = []
    # 2D array with indicies: [ (x0,y0), (x1,y1) ]
    if len(c[0]) == 2:
        subImage = image[c[0][0]:c[1][0], c[0][1]:c[1][1]]
        if subImage.size == 0:
            print(c)
            print("this is empty")
        
        
    # 3D array with indicies: [ (z0,y0,x0), (z1,y1,x1) ]
    if len(c[0]) == 3 & c[0][0] == c[1][0]:
        # print("3d roi")
        subImage = image[c[0][0]:c[1][0], c[0][1]:c[1][1], c[0][2]:c[1][2]]
        subImage = subImage[0, :, :]
        
    return subImage

def gen_features(coord, args):
    
    diam = args['diam']
            
    rect = [(coord[0] - diam, coord[1] - diam), (coord[0] + diam, coord[1] + diam)]
    # rect = [(coord[0], coord[1]), (coord[0] + diam, coord[1] + diam)]
    
    if rect[0][0]<0 or rect[0][1]<0 or rect[1][0]<0 or rect[1][1]<0:
        print("rect bad")
        print(rect)

    subImage = extractRect(args['image'], rect)
    if subImage.size == 0:
        print("Empty")
        return
    # Initialize the feature list. These entries
    # provide info on the patient, image and ROI that produced the features
    # use a dumb trick to make some features appear first in the sorted output
    features = {}
    # features["0_Image Filename"] = args["filename"]
    # features["0_Image Contrast"] = args["contrast"]
    # tags = ["0_patientID", "0_image name", "0_mask", "0_slice", "0_image acquisition date"]
    # for tag in tags:
    #     features[tag] = args[tag]
                
    features["Y"]   = coord[0]
    features["X"]   = coord[1] 
            
            
    features["Mean"]   = subImage.mean()
    features["Range"]  = subImage.ptp()
    features["StdDev"] = subImage.std()
    features["Kurtosis"] = stats.kurtosis(subImage.flatten())
    features["Skewness"] = stats.skew(subImage.flatten())
        
    features['modality'] = args['modality']
        
            
    # subImage_copy = np.array(subImage, copy=True)
    # subImage_scaled = scaleIntensity(subImage_copy, 0, args["grayscales"])
    # subImage_int = np.rint(subImage_scaled).astype(np.uint8)
            
    # for i,name in enumerate(args["algos"]):
    #     features.update(args["algos"][i](subImage_int))
                
    return features


def saveTabularData(imageObservations, csv_writer):
    '''
    Save data into format where each observation is a window with features describing mean,range, skew,kurtosis,std for each image type available to that date
    '''

    data = {}
    # imageObservation is a list of list of dicts
    # window_data is a list of dicts
    for window_data in imageObservations:
        # window is a dict in the list of dicts (window_data)
        for window in window_data:
            coords = "{},{},{}".format(window['X'], window['Y'], window['Z'])
            if data.get(coords) is None:
                data[coords] = {}

            modality = window['modality']

            data[coords]['X'] = window['X']
            data[coords]['Y'] = window['Y']
            data[coords]['Z'] = window['Z']
            data[coords]['mean_{}'.format(modality)] = window['Mean']
            data[coords]['std_{}'.format(modality)] = window['StdDev']
            data[coords]['kurtosis_{}'.format(modality)] = window['Kurtosis']
            data[coords]['skew_{}'.format(modality)] = window['Skewness']
            data[coords]['range_{}'.format(modality)] = window['Range']
            data[coords]['patient'] = window['patient']
            data[coords]['sex'] = window['sex']
            data[coords]['date'] = window['date']

    for coords in data:
        csv_writer.writerow([data[coords]['patient'], data[coords]['date'], data[coords]['sex'], data[coords]['X'], data[coords]['Y'], data[coords]['Z'],coords, 
                                data[coords].get('mean_T1'), data[coords].get('std_T1'), data[coords].get('skew_T1'), data[coords].get('kurtosis_T1'), data[coords].get('range_T1'),
                                data[coords].get('mean_T1Gd'), data[coords].get('std_T1Gd'), data[coords].get('skew_T1Gd'), data[coords].get('kurtosis_T1Gd'), data[coords].get('range_T1Gd'),
                                data[coords].get('mean_T2'), data[coords].get('std_T2'), data[coords].get('skew_T2'), data[coords].get('kurtosis_T2'), data[coords].get('range_T2'),
                                data[coords].get('mean_FLAIR'), data[coords].get('std_FLAIR'), data[coords].get('skew_FLAIR'), data[coords].get('kurtosis_FLAIR'), data[coords].get('range_FLAIR')])


def saveFeatures(features, csv_writer):
    '''
    Take all the features generated from an image slice and save them to csv
    '''

    for window in features:
        csv_writer.writerow([window['patient'], window['date'], window['sex'], 
                            window['X'], window['Y'], window['Z'], 
                            window['modality'], window['Mean'], window['StdDev'],
                             window['Skewness'], window['Kurtosis'], window['Range']])



def extractFeatures(image, brainMask, csv_writer):
    '''
    Compute features for each slice in the MRI image and then save them to disk
    Features currently generated include: mean, standard deviation, skew, kurtosis, range
    '''

    # set number of cores for parallelization and sliding window size
    CORES = 16
    GRID_SPACING = 8

    # set up multiprocessing to parallel process 1 image with multiple threads
    # use manager to share memory for all the subprocessses
    pool = mp.Pool(CORES)
    manager = mp.Manager()
    gen_features_args = manager.dict()
    gen_features_args['diam'] = GRID_SPACING
    gen_features_args['modality'] = ImageSubtype.get_modality(image.image_subtype.id)
    patient = image.patient

    # get numpy represntations of brainmask and the image
    imageMatrix = sitk.GetArrayFromImage(image.sitk_image).T.astype(np.double)
    brainMaskMatrix = sitk.GetArrayFromImage(brainMask).T.astype(np.double)

    # For Testing
    # Image.snapshow([image.sitk_image], segmentation=brainMask)

    # use the brainMask to figure out which coordinates of the image are actually brain and should be computed
    # for each slice in the brainmask, calculate the coordinates for that slice and store the list in coordSlices
    coordSlices = []
    # sub_region = []
    slices = brainMaskMatrix.shape[2]
    # slices = image.shape[2]
    for index in range(slices):
        brainMaskSlice = brainMaskMatrix[:,:, index]
        # roiSlice = image[:,:, index]
        coords = []
        for x in range(brainMaskSlice.shape[0]): 
            if x % GRID_SPACING != 0:
                continue
            for y in range(brainMaskSlice.shape[1]):
                if y % GRID_SPACING == 0 and brainMaskSlice[x,y]:
                    coords.append((x,y))
                    # coords.append((x,y,index))
                    # case.append(caseName)
                    # if roiSlice[x,y] == 1:
                    #     sub_region.append((1,0,0,0))
                    # elif roiSlice[x,y] == 2:
                    #     sub_region.append((0,1,0,0))
                    # elif roiSlice[x,y] == 4:
                    #     sub_region.append((0,0,1,0))
                    # elif roiSlice[x,y] == 8:
                    #     sub_region.append((0,0,0,1))
                    # elif roiSlice[x,y] == 3:
                    #     sub_region.append((1,1,0,0))
                    # elif roiSlice[x,y] == 5:
                    #     sub_region.append((1,0,1,0))
                    # elif roiSlice[x,y] == 9:
                    #     sub_region.append((1,0,0,1))
                    # elif roiSlice[x,y] == 7:
                    #     sub_region.append((1,1,1,0))
                    # elif roiSlice[x,y] == 11:
                    #     sub_region.append((1,1,0,1))
                    # elif roiSlice[x,y] == 13:
                    #     sub_region.append((1,0,1,1))
                    # elif roiSlice[x,y] == 14:
                    #     sub_region.append((0,1,1,1))
                    # elif roiSlice[x,y] == 15:
                    #     sub_region.append((1,1,1,1))
                    # elif roiSlice[x,y] == 6:
                    #     sub_region.append((0,1,1,0))
                    # elif roiSlice[x,y] == 10:
                    #     sub_region.append((0,1,0,1))
                    # elif roiSlice[x,y] == 12:
                    #     sub_region.append((0,0,1,1))
        coordSlices.append(coords)
    
    sliceCount = imageMatrix.shape[2]
    print(sliceCount)
    observations = []
    for index in range(sliceCount):
        if coordSlices[index] == []:
            continue
        mriSlice = imageMatrix[:,:,index]
        gen_features_args['image'] = mriSlice
        partial_gen_features = partial(gen_features, args=gen_features_args)

        features = pool.map(partial_gen_features, coordSlices[index])
        for feature in features:
            feature['Z'] = index
            feature['patient'] = patient.patient_number
            feature['sex'] = patient.sex
            feature['date'] = image.image_date
            # feature['ROI'] = sub_region
        
        saveFeatures(features, csv_writer)

        observations += features
    pool.close()
    
    return observations

def processData(cases):
    '''
    Feature Generation Process:
        get dates and associated ids
        load ids into images, get masks and normalize
        for each image in date:
            calculate indexing with the mask
            extract features with mask - run in parallel
            write features to csv file 
    '''
    

    # if output files don't exist create new ones - else just append to current ones
    # NOTE: This means if you want to start a new run you should delete the feature_data.csv and tabular_data.csv files already present
    if os.path.isfile(os.path.join(dataDirectory, featureFileName)) is False:
        featureFile = open(os.path.join(dataDirectory, featureFileName), 'w')
        csv_writer = csv.writer(featureFile, delimiter=',')
        csv_writer.writerow(['patient', 'date', 'sex', 'X', 'Y', 'Z', 'modality', 'mean', 'std',  'skew', 'kurtosis','range'])
    else:
        featureFile = open(os.path.join(dataDirectory, featureFileName), 'a')
        csv_writer = csv.writer(featureFile, delimiter=',')

    
    if os.path.isfile(os.path.join(dataDirectory, tabularFileName)) is False:
        tabularFile = open(os.path.join(dataDirectory, tabularFileName), 'w')
        tabular_writer = csv.writer(tabularFile, delimiter=',')
        tabular_writer.writerow(['patient', 'date', 'sex', 'X', 'Y', 'Z', 'coordinates', 
                            'mean_T1', 'std_T1','skew_T1','kurtosis_T1','range_T1',
                            'mean_T1Gd','std_T1Gd','skew_T1Gd','kurtosis_T1Gd','range_T1Gd',
                            'mean_T2','std_T2','skew_T2','kurtosis_T2','range_T2',
                            'mean_FLAIR','std_FLAIR','skew_FLAIR','kurtosis_FLAIR','range_FLAIR'])
    else:
        tabularFile = open(os.path.join(dataDirectory, tabularFileName), 'a')
        tabular_writer = csv.writer(tabularFile, delimiter=',')
    

    # Skip to case containing this id. Only will process this case and any that follow
    # NOTE: initial ids associated with cases are not the registered image ids. Therefore must use base ID as start ID
    startId = None

    if startId is not None:
        for case in cases:
            if startId in case['ids']:
                index = cases.index(pairing)
                break
        cases = cases[index:]
        print("Skipped {} cases".format(index))


    
    for case in cases:
        logging.info("Extracting features from patient {} date {}".format(case['patient'], case['date']))
        ids = case['ids']
        images = loadImages(ids, date=case['date'])

        # if images[0].matrix.shape[2] >= 60:
        #     logging.warning("Image size too big for analysis for patient {} date {} SKIPPING".format(case['patient'], case['date']))
        #     continue

        logging.info("Images finished loading")
        for im in images:
            if im.id == case['base_id']:
                filePath = '/PatientImages/{}/{}/BRAIN_MASK.nii.gz'.format(im.patient.patient_number, im.image_date.strftime('%Y-%m-%dT%H-%M-%S+00-00'))
         
                brainMask = sitk.ReadImage(filePath)
                _, brainMask = Image.resampleImageSpacing(im.sitk_image, (1.0547, 1.0547, 0), brainMask)

        if images is None or brainMask is None:
            logging.critical("NO BRAIN MASKS COULD BE LOADED for patient {} date {}. SKIPPING".format(case['patient'], case['date']))
            continue

        image_observations = []

        for image in images:
            logging.info("Extracting features from image {}".format(image.id))
            results = extractFeatures(image, brainMask, csv_writer)
            image_observations.append(results)
            logging.info("Finished feature extraction for {} id {}".format(ImageSubtype.get_modality(image.image_subtype.id), image.id))
        
        logging.info("Saving data in tabular format for patient {} date {}".format(case['patient'], case['date']))
        saveTabularData(image_observations, tabular_writer)
        logging.info("Finished saving tabular data for patient {} date {}".format(case['patient'], case['date']))
        
        print("{} Finished".format(case['date']))
        logging.info("{} Finished".format(case['date']))

    featureFile.close()
    tabularFile.close()
    

def featureExtraction():
    logging.basicConfig(filename=os.path.join(mainDirectory, logFilePath),level=logging.DEBUG, 
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('START RUN {}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))

    cases = loadCases()
    processData(cases)

    logging.info('END RUN {}'.format(strftime("%Y-%m-%d %H:%M:%S", localtime())))

def joinFeaturesROI(roi_data):
    TabularData = pd.read_csv(os.path.join(dataDirectory,tabularFileName))  

    TabularData['case'] = TabularData['patient'] + '_' + TabularData['date']

    # df.apply(lambda row: label_race(row), axis=1)

    # final_TabularData = pd.merge(roi_data, TabularData, how='right', on='coordinates')
    print('-- Tabular Data')
    print(TabularData.head())
    print('-- ROI Data')
    print(roi_data.head())

    final_TabularData = pd.merge(TabularData, roi_data, how='left', on = ['case', 'coordinates'])

    print(final_TabularData.head())

    final_TabularData['ROI'] = final_TabularData['ROI'].fillna(False)
    final_TabularData.loc[final_TabularData['ROI']==False, 'ROI'] = 0
    final_TabularData.loc[final_TabularData['ROI']==True, 'ROI'] = 1

    print(final_TabularData.head())

    final_TabularData.to_csv(os.path.join(dataDirectory, roiFileName), index=False)

    # return final_TabularData

def main():

    # TabularData = pd.read_csv(os.path.join(dataDirectory,'tabular_data.csv'))  

    # TabularData['case'] = TabularData['patient'] + '_' + TabularData['date']

    # print(TabularData.head())

    # featureExtraction()
    roi_data = tumorRegion()
    joinFeaturesROI(roi_data)
    
    

if __name__ == '__main__':
    main()
