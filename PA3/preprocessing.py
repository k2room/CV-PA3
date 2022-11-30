import os
import cv2
from tqdm import tqdm
import shutil

if __name__=='__main__':
    path = '/home/work/CV-PA3/PA3/dataset'
    save_path = '/home/work/CV-PA3/PA3/dataset_processing/'
    save_size = (512, 512)

    with open(os.path.join(path, 'IHD_train.txt'), 'r') as f1:
        train_list = f1.readlines()
        train_list = [i.strip() for i in train_list]

    with open(os.path.join(path, 'IHD_test.txt'), 'r') as f2:
        test_list = f2.readlines()
        test_list = [i.strip() for i in test_list]
    
    name = ['HAdobe5k', 'HCOCO', 'Hday2night', 'HFlickr']

    for i in name:
        path_folder = os.path.join(path, i)
        save_path_folder = os.path.join(save_path, i)
        # shutil.copyfile(os.path.join(path_folder,i+'_test.txt'),save_path_folder)
        # shutil.copyfile(os.path.join(path_folder,i+'_train.txt'),save_path_folder)
        os.makedirs(save_path_folder, exist_ok=True)
        os.makedirs(os.path.join(save_path_folder, 'composite_images'), exist_ok=True)
        os.makedirs(os.path.join(save_path_folder, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_path_folder, 'real_images'), exist_ok=True)

    all_list = train_list+test_list
    print("IHD_train :", len(train_list), " |  IHD_test :", len(test_list))
    for img in tqdm(all_list):
        name_parts = img.split('_')
        comp_path = img
        save_comp_path = os.path.join(save_path, comp_path)
        if os.path.exists(save_comp_path):
            continue

        mask_path = img.replace('composite_images','masks')
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png')
        save_mask_path = os.path.join(save_path, mask_path)

        real_path = img.replace('composite_images','real_images')
        real_path = real_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
        save_real_path = os.path.join(save_path, real_path)

        # image read and resize
        comp = cv2.imread(os.path.join(path, comp_path))
        comp = cv2.resize(comp, dsize = save_size, interpolation = cv2.INTER_CUBIC)
        mask = cv2.imread(os.path.join(path, mask_path))
        mask = cv2.resize(mask, dsize = save_size, interpolation = cv2.INTER_NEAREST)
        real = cv2.imread(os.path.join(path, real_path))
        real = cv2.resize(real, dsize = save_size, interpolation = cv2.INTER_CUBIC)

        cv2.imwrite(save_comp_path, comp)
        cv2.imwrite(save_mask_path, mask)
        cv2.imwrite(save_real_path, real)

    f1.close()
    f2.close()