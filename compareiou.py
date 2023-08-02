import sys

import os
path = 'testoutputs'
#path = 'skipframeoutput'
import glob
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import time


experiment_no = 8
save_data = False
# 1: optical flow 
# 2: optical flow of 5 frames
# 3: between the real mask t and t+1
# 4: average optical flow
# 5: optical flow for average mask

if experiment_no == 1:
    path = 'testoutputs'
elif experiment_no == 2:
    path = 'skipframeoutput'
elif experiment_no == 3:
    path = 'testoutputs' # doesnot matter here
elif experiment_no == 4:
    path = 'averageflows'
elif experiment_no == 5:
    path ='averageflows'
elif experiment_no == 6:
    path = 'largeoutput'
elif experiment_no == 7:
    path = 'largeoutput'
elif experiment_no == 8:
    path = 'vmdresults'

def calculate_iou(mask1, mask2):
    # Convert masks to binary (0 or 1)
    # mask1_binary = (mask1 > 0.5).astype(np.uint8)
    # mask2_binary = (mask2 > 0.5).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Compute IOU
    iou = np.sum(intersection) / np.sum(union)

    return iou

# read all folders in testouts
#for each video in dataset, we need a new path
directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
print(directories)
#print(directories)




# for each folder load predicted masks and load real mask
# get the folder name, 
count = 0
totaliou = 0

# for pandas, aka data analysis

# frame level
ious = []
frameno = []
video = []

# video level
vious = []
framestotal = []
videoid = []



for dir in directories:
    
    print(dir)
    videoiou = 0
    vidcount = 0
    # if dir != 'testing2':
    #     continue
    maskpath = os.path.join(path, dir)
    

    realmask = os.path.join('test', dir)
    realmask = os.path.join(realmask, 'SegmentationClassPNG')
    #print(realmask, maskpath)


    preds = glob.glob(os.path.join(maskpath, '*.jpg'))
    reals = glob.glob(os.path.join(realmask, '*.png'))

    #print(len(preds), len(reals))
    f1 = None
    f2 = None

    # 1: optical flow 
    # 2: optical flow of 5 frames
    # 3: between the real mask t and t+1
    # 4: average optical flow
    # 5: just between the masks for testing
    # 6: average optical flow on all data, 60
    # 7: beteen the real mask t and t+1 but on all dataset
    # 8: between vmd dataset and real, for testing its performance

    
    if experiment_no == 1:
        f1 = reals[1:]
        f2 = preds[:-1]
    elif experiment_no == 2:
        f1 = reals[5:]
        f2 = preds[:-1]
    elif experiment_no == 3:
        f1 = reals[1:]
        f2 = reals[:-1]
    elif experiment_no == 4:
        f1 = reals[1:]
        f2 = preds[:-1]
    elif experiment_no == 5:
        f1 = reals[1:]
        f2 = preds[:-1]
    elif experiment_no == 6:
        f1 = reals[1:]
        f2 = preds[:-1]
        
    elif experiment_no == 7:
        f1 = reals[1:]
        f2 = reals[:-1]
    elif experiment_no == 8:
        f1 = reals[:]
        preds = glob.glob(os.path.join(maskpath, '*.png'))
        f2 = preds[:]
        



    
    
    for pred, real in zip(f2, f1):
        count += 1
        #print(pred, real, type(pred))

        predimage = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)

        # Convert to binary mask (0 or 1)
        predimage = (predimage > 0).astype(np.uint8)

        realimage = cv2.imread(real, cv2.IMREAD_GRAYSCALE)

        # Convert to binary mask (0 or 1)
        realimage = (realimage > 0).astype(np.uint8)


        iou = calculate_iou(predimage, realimage)
        totaliou += iou
        videoiou += iou
        vidcount += 1
        #print(iou)

        # ious = []
        # frameno = []
        # video = []

        temp = pred.split('\\')
        ious.append(iou)
        frameno.append(temp[-1])
        video.append(temp[-2])
    
    #print(frameno, ious, video)


    try:
        print('average of this clip', videoiou/vidcount, vidcount, dir)
        print()
    except:
        print('error in dir', dir, vidcount)
        print()


    if vidcount != 0:
        vious.append(videoiou/vidcount)
        framestotal.append(vidcount)
        videoid.append(dir)
    
#print(vious, framestotal, videoid)

# creating dataframe and saving it for videos
data = {'video': videoid,
        'frames': framestotal,
        'iou': vious}
videos = pd.DataFrame(data)
videos.to_csv('videoious.csv')

# creating dataframe for individual frames 
framesdata = {
    'video': video, 
    'frameno': frameno,
    'iou': ious
}

frames_df = pd.DataFrame(framesdata)
frames_df.to_csv('frames_iou.csv')

print('average iou', totaliou/count, count)



      



    

# read them as binary images

# zip them together, and compute iou score using jaccard similarity, which is the same as og iou


