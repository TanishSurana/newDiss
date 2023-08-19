from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from PIL import Image


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    #ax.imshow(img)
    return img

# detectron
to_test = {'MSD': "test/"}


def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]

args = {
    'scale': 384,
    'test_adjacent': 1,
    'input_folder': 'JPEGImages',
    'label_folder': 'SegmentationClassPNG',
    'crf': True
}




def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
def main():
    sam_checkpoint = "C:\\Users\\Tanish\\Desktop\\Tanish Dissertation\\newDiss\\segment-anything-main\\segment-anything-main\\sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    root ='C:\\Users\\Tanish\\Desktop\\Tanish Dissertation\\newDiss\\semi_optic_features\\test'

    
    video_list = listdirs_only(os.path.join(root))

    for video in tqdm(video_list):

        # all images
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video)) if
                    f.endswith('.jpg')]
        
        img_eval_list = sortImg(img_list)
        print(img_eval_list)

        for exemplar_idx, exemplar_name in enumerate(img_eval_list):
            

            image = cv2.imread(os.path.join(root, video, exemplar_name + '.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            #img = Image.open(os.path.join(root, video, exemplar_name + '.jpg')).convert('RGB')
            masks = mask_generator.generate(image)
            print(masks, 'mask print')
            final = show_anns(masks)
            print(final)
      
        
            check_mkdir(os.path.join("results", video))
            save_name = f"{exemplar_name}.png"
            #print(os.path.join("results", video, save_name))
            Image.fromarray(final).save(os.path.join("results", video, save_name))









if __name__ == '__main__':
    main()
