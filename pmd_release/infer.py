import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import skimage.io
from config import msd_testing_root
from misc import check_mkdir, crf_refine

from model.pmd import PMD
device_ids = [0]
torch.cuda.set_device(device_ids[0])

from tqdm import tqdm


args = {
    'scale': 384,
    'crf': True
}

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# detectron
to_test = {'MSD': msd_testing_root}
to_test = {'MSD': "test/"}

to_pil = transforms.ToPILImage()

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

def main():
    net = PMD().cuda(device_ids[0])
    net.load_state_dict(torch.load('pmd_splited.pth'))
    
    net.eval()

    root ='../fulldataset/test'

    with torch.no_grad():
        video_list = listdirs_only(os.path.join(root))

        for video in tqdm(video_list):

            # all images
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['input_folder'],)) if
                        f.endswith('.jpg')]
            # need evaluation images
            img_eval_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, args['label_folder'])) if f.endswith('.png')]

            img_eval_list = sortImg(img_eval_list)

            for exemplar_idx, exemplar_name in enumerate(img_eval_list):
            
                img = Image.open(os.path.join(root, video, args['input_folder'], exemplar_name + '.jpg')).convert('RGB')


                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                # f_4, f_3, f_2, f_1, e = net(img_var)
                f_4, f_3, f_2, f_1, edge, final = net(img_var)
                # output = f.data.squeeze(0).cpu()
                # edge = e.data.squeeze(0).cpu()
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                edge = edge.data.squeeze(0).cpu()
                final = final.data.squeeze(0).cpu()

                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                final = np.array(transforms.Resize((h, w))(to_pil(final)))

                if args['crf']:
                    final = crf_refine(np.array(img.convert('RGB')), final)

                # Image.fromarray(final).save(os.path.join("result", name, img_name[:-4] + ".png"))

                check_mkdir(os.path.join("results", video))
                save_name = f"{exemplar_name}.png"
                #print(os.path.join("results", video, save_name))
                Image.fromarray(final).save(os.path.join("results", video, save_name))




        

        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                check_mkdir(os.path.join("result", name))

                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                # f_4, f_3, f_2, f_1, e = net(img_var)
                f_4, f_3, f_2, f_1, edge, final = net(img_var)
                # output = f.data.squeeze(0).cpu()
                # edge = e.data.squeeze(0).cpu()
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                edge = edge.data.squeeze(0).cpu()
                final = final.data.squeeze(0).cpu()

                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                final = np.array(transforms.Resize((h, w))(to_pil(final)))

                if args['crf']:
                    final = crf_refine(np.array(img.convert('RGB')), final)

                Image.fromarray(final).save(os.path.join("result", name, img_name[:-4] + ".png"))

            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
