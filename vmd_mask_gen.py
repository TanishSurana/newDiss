
# OVERVIEW OF COMPARISION
# we have mt and mt+1 masks
# we have it and it+1 images (frames)

# use it and it+1 images to generate optical flow ot
# ot will tell us how the pexels are moving in form of vectors between it and it+1. 
# now we can use this ot and select only the pixels in mask mt, to predict the next mask: m*t+1

# this m*t+1 in theory should be equivalent to mt+1.

# we will compare mt+1 and m*t+1, to see how good this or any optical flow information is in predicting mask aka mirror movement. 

# to compare basic will be IoU, Pixel accuracy, F1 score
# to compare advance stuff will be boundary based metrics: average symmetic surface distance ASSD, and Hausdorff Distance. 

import sys
sys.path.append('RAFT/core')



import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder, coords_grid, bilinear_sampler
import time
from tqdm import tqdm, trange


# TODO: first we need to generate optical flow, ot+1 using RAFT model, this will use input as it and it+1
# first we need an optical flow model: RAFT


# working on full dataset, size idk lets see


# a better work flow would be to use all the images at once, upsides easy processing, downside we cannot use this functions created here anywhere, if single image then these will be useful...
# lets to batch or whole dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def averagewrap(im1, flow12):
    """
    Warp an image (im1) to image2, according to the optical flow (flow12).
    im1: [B, C, H, W] (image1)
    flow12: [B, 2, H, W] (optical flow from image1 to image2)
    """
    B, C, H, W = im1.size()

    # Calculate the average flow vector
    average_flow_vector = flow12.mean(dim=(2, 3))

    # Repeat the average flow vector to match the dimensions of the image
    average_flow_vector = average_flow_vector.view(B, 2, 1, 1).repeat(1, 1, H, W)

    average_flow_vector = average_flow_vector.to('cpu')

    # Generate the grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).float()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).float()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

    # Add the average flow vector to the grid
    grid = torch.cat((xx, yy), dim=1)
 
   
    grid = grid - average_flow_vector

    if im1.is_cuda:
        grid = grid.cuda()

    # Normalize the grid values to [-1, 1]
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)
    warped_im1 = F.grid_sample(im1, grid)

    return warped_im1


def warp2(im1, flow12):
    """
    Warp an image (im1) to image2, according to the optical flow (flow12).
    im1: [B, C, H, W] (image1)
    flow12: [B, 2, H, W] (optical flow from image1 to image2)
    """
    B, C, H, W = im1.size()

    # Generate the grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).float()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).float()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), dim=1)

    if im1.is_cuda:
        grid = grid.cuda()

    # Warp image1 to image2
    vgrid = grid + flow12
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    warped_im2 = F.grid_sample(im1, vgrid)

    return warped_im2

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output

def warpseg(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [H, W, C] (im2)
    flo: [B, 2, H, W] flow
    """
    print('here', x.shape, flo.shape)
    H, W, C = x.size()
    # we need to change the shape of x so that flow and x have the same shape

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W).repeat(B, 1, 1)
    yy = yy.view(1, H, W).repeat(B, 1, 1)
    grid = torch.cat((xx, yy), 0).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[0, :, :] = 2.0 * vgrid[0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[1, :, :] = 2.0 * vgrid[1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(1, 2, 0)
    output = F.grid_sample(x, vgrid.unsqueeze(0))
    mask = torch.ones(x.size())
    mask = F.grid_sample(mask, vgrid.unsqueeze(0))

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image_proper(imfile):
    img = np.array(cv2.imread(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



def opticflow(args, path, videoresultpath):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad(): # as we are interfering not training
        images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))
        
        images = sorted(images)
        outputs = []
        outid = []
    
        with tqdm(total=len(images)) as pbar:

            for imfile1, imfile2 in zip(images[:-1], images[1:]):

                image1 = load_image_proper(imfile1)
                image2 = load_image_proper(imfile2)



                frameno = imfile1.split('\\')[-1] # saving it as optical flow between 1 and 2 image, so imfile 2 == 000_2.png

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                _, flow12 = model(image1, image2, iters=24, test_mode=True)

        

                #print(image1.shape, type(image1))


                # adapted to generate flows
                # maskpath1 = imfile1.replace('JPEGImages', 'SegmentationClassPNG') # this is the path of the mask of 1st image
                # maskpath1 = maskpath1.replace('jpg', 'png') # this is the path of the mask of 1st image
                # segmentation_mask = load_image_proper(maskpath1)

                # out = warp2(image1, flow12)
                # out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # out = (out*1).astype(np.uint8)

            
            
                #segmentation_mask =  cv2.imread(maskpath1) # reading as a gray scale
                
                #print('shape of segmentation mask',segmentation_mask.shape)
                

                # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                # flow_up = flow_up[0].permute(1,2,0).cpu().numpy()

                # h = flow_up.shape[0]
                # w = flow_up.shape[1]
                # flow_up[:,:,0] += np.arange(w)
                # flow_up[:,:,1] += np.arange(h)[:,np.newaxis]
                # # img1 = load_image_proper(imfile1)
                # img2 = load_image_proper(imfile2)
                # warped_img2 = cv2.remap(img1, flow_up, None, cv2.INTER_LINEAR)
                #plt.imshow(image1.permute(1, 2, 0))
                




                smallpath = str(frameno)
        

                # this if for images only we need files

                # result=cv2.imwrite(os.path.join(videoresultpath, smallpath), out)
                # if result==True:
                #     print("File saved successfully")
                # else:
                #     print("Error in saving file")

                # out = averagewrap(segmentation_mask, flow12) # created movement in segmentation mask
                # out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # out = (out*1).astype(np.uint8)

                # outputs.append(out)
                # outid.append(smallpath)

        
                pbar.update(int(100/len(images)))




                
                #CODE FOR DISPLAYING IMAGE OF OPTICAL FLOW
                flo = flow12[0].permute(1,2,0).cpu().numpy()
                
                # map flow to rgb image
                flo = flow_viz.flow_to_image(flo)
                outputs.append(flo)
                outid.append(smallpath)

        

                # cv2.imshow('image', flo)
                # cv2.waitKey()
                
                # smallpath = str(count) + 'hehe.png'
                
                # this if for images only we need files
                #savepath = 'testwarp'


                # import matplotlib.pyplot as plt
                # plt.imshow(img_flo / 255.0)
                # plt.show()

                # THIS IS TO SHOW OPTICAL FLOW
                # print(flow12)

                # flo = flow12[0].permute(1,2,0).cpu().numpy()
                
                # # map flow to rgb image
                # flo = flow_viz.flow_to_image(flo)


                # import matplotlib.pyplot as plt
                # plt.imshow(img_flo / 255.0)
                # plt.show()

        total_outputs = len(outputs)
        for i in range(total_outputs):
            result=cv2.imwrite(os.path.join(videoresultpath, outid[i]), outputs[i])
            if result==True:
                pass
                #print("File saved successfully", smallpath)
            else:
                print("Error in saving file", videoresultpath, smallpath)





# run code
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = 'models/raft-things.pth'
    #path = 'fulldataset/train'
    path = 'test'

    #for each video in dataset, we need a new path
    directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    #print(directories)

    # Get the full path of each subdirectory
    subdirectory_paths = [os.path.join(path, directory) for directory in directories]

    # Print the list of subdirectory paths
    for subdirectory_path in subdirectory_paths:
        subsub = [name for name in os.listdir(subdirectory_path) if os.path.isdir(os.path.join(subdirectory_path, name))]
        # this if else statement below, checks if the name of the directory are in proper order, if not it willl reverse it. 
        if 'Segmentation' in subsub[0]:
            images = os.path.join(subdirectory_path, subsub[1])
            masks = os.path.join(subdirectory_path, subsub[0])
        else:
            images = os.path.join(subdirectory_path, subsub[0])
            masks = os.path.join(subdirectory_path, subsub[1])

        #print(images, 'here is the supposed jpegimages folder name or whatever', type(images))

        # making the directories to store the images
        videofolder = images.split('\\')[1]
        #parent = 'testoutputs/'
        #parent = 'skipframeoutput/'
        parent = 'opticalflowfeatures/'
        testpath = os.path.join(parent, videofolder)
        #print(testpath)

        try: 
            os.mkdir(testpath) 
            print('created', testpath)
        except OSError as error: 
            print(error)

        # now we have made the folder for each video, now we need to store the output masks there. 


        opticflow(args, images, testpath)
        print()

        # now we have all images and masks for 1 video
        # now pass images to optical flow, that will give us optical flow



        # with all optical flow, we need to only calculate




    
        

    

    #demo(args)
