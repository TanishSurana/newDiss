import os
import shutil

def list_files(directory):
    """List all files in a given directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def list_folders(directory):
    """List all folders in a given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def check_make_dir(directory_path):


    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


root = {
    'mask': 'fulldataset',
    'optic': 'sam_optic_mask',
    'premask': 'vmd_masks'
}


for k, v in root.items():
    #print(k,v)
    vvtest = os.path.join(v,'test')
    vvtrain = os.path.join(v,'train')



    test = list_folders(vvtest)
    train = list_folders(vvtrain)


    test_count = 0
    train_count = 0

    if v == 'fulldataset':



        for folder in test:
            folder_path = os.path.join(vvtest, folder, 'SegmentationClassPNG')
            # now saving it for each file
            #print(folder, folder_path)
            files = list_files(folder_path)
            mainfolder = 'all' + v 
            folder2 = os.path.join(mainfolder, 'test')
            check_make_dir(folder2)
            for f in files:
                file_path = os.path.join(folder_path, f)
                #print(file_path)

                fff = str(test_count) + f

                rename = os.path.join(mainfolder, 'test', fff)
                #print(rename)
                test_count += 1

                shutil.copy2(file_path, rename)

        for folder in train:
            folder_path = os.path.join(vvtrain, folder, 'SegmentationClassPNG')
            # now saving it for each file
            files = list_files(folder_path)
            mainfolder = 'all' + v 
            folder2 = os.path.join(mainfolder, 'train')
            check_make_dir(folder2)
            for f in files:
                file_path = os.path.join(folder_path, f)
                #print(file_path)

                fff = str(train_count) + f

                rename = os.path.join(mainfolder, 'train', fff)
                #print(rename)
                train_count += 1

                shutil.copy2(file_path, rename)
