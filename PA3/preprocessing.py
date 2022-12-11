import os
import cv2
from tqdm import tqdm
import random
import shutil

if __name__=='__main__':
    path = '/home/work/CV-PA3/PA3/dataset'
    save_path = '/home/work/CV-PA3/PA3/dataset_processing/'
    save_size = (512, 512)
    augmentation = True
    num_aug = 3

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

        if augmentation:
            save_path_folder_aug = os.path.join(save_path, (i+'_aug'))
            os.makedirs(save_path_folder_aug, exist_ok=True)
            os.makedirs(os.path.join(save_path_folder_aug, 'composite_images'), exist_ok=True)
            os.makedirs(os.path.join(save_path_folder_aug, 'masks'), exist_ok=True)
            os.makedirs(os.path.join(save_path_folder_aug, 'real_images'), exist_ok=True)

    # all_list = train_list+test_list
    all_list = train_list
    print("IHD_train :", len(train_list), " |  IHD_test :", len(test_list))
    for img in tqdm(all_list): 
        name_parts = img.split('_') 

        comp_path = img # ex: HAdobe5k/composite_images/a3630_1_5.jpg
        save_comp_path = os.path.join(save_path, comp_path)

        mask_path = img.replace('composite_images','masks') # comp : index2_1_1.jpg, index2_1_2.jpg -> mask : index2_1.jpg
        mask_path = mask_path.replace(('_'+name_parts[-1]),'.png') # ex: HAdobe5k/masks/a3630_1.png
        save_mask_path = os.path.join(save_path, mask_path)

        real_path = img.replace('composite_images','real_images') # comp : index2_1_1.jpg, index2_1_2.jpg -> real : index2.jpg
        real_path = real_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg') # ex: HAdobe5k/real_images/a3630.jpg
        save_real_path = os.path.join(save_path, real_path)

        aug_parts = img.split('/')
        comp_aug_path = aug_parts[0]+'_aug'+'/'+aug_parts[1]+'/'+aug_parts[2] # ex: HAdobe5k_aug/composite_images/a3630_1_5.jpg
        aug_parts = comp_aug_path.split('.') 
        aug_parts2 = comp_aug_path.split('_') 
        
        if augmentation:
            save_comp_aug_path = os.path.join(save_path, comp_aug_path)
            
            if os.path.exists(save_comp_aug_path):
                continue

            mask_aug_path = comp_aug_path.replace('composite_images','masks')
            # mask_aug_path = mask_aug_path.replace(('_'+aug_parts2[-1]),('_aug'+str(i)+'.png'))
            mask_aug_path = mask_aug_path.replace(('_'+name_parts[-1]),'.png')
            save_mask_aug_path = os.path.join(save_path, mask_aug_path)

            real_aug_path = comp_aug_path.replace('composite_images','real_images')
            # real_aug_path = real_aug_path.replace(('_'+aug_parts2[-2]+'_'+aug_parts2[-1]),('_aug'+str(i)+'.jpg'))
            real_aug_path = real_aug_path.replace(('_'+name_parts[-2]+'_'+name_parts[-1]),'.jpg')
            save_real_aug_path = os.path.join(save_path, real_aug_path)

            aug_comp = cv2.imread(os.path.join(path, comp_path))
            aug_mask = cv2.imread(os.path.join(path, mask_path))
            aug_real = cv2.imread(os.path.join(path, real_path))
            
            aug_comp_i = cv2.resize(aug_comp, dsize = save_size, interpolation = cv2.INTER_CUBIC)
            aug_mask_i = cv2.resize(aug_mask, dsize = save_size, interpolation = cv2.INTER_NEAREST)
            aug_real_i = cv2.resize(aug_real, dsize = save_size, interpolation = cv2.INTER_CUBIC)

            i = random.randint(1,num_aug)

            if i%num_aug == 1:
                aug_comp_i = cv2.rotate(aug_comp_i, cv2.ROTATE_90_CLOCKWISE)
                aug_mask_i = cv2.rotate(aug_mask_i, cv2.ROTATE_90_CLOCKWISE)
                aug_real_i = cv2.rotate(aug_real_i, cv2.ROTATE_90_CLOCKWISE)
            elif i%num_aug == 2:
                aug_comp_i = cv2.flip(aug_comp_i, -1)
                aug_mask_i = cv2.flip(aug_mask_i, -1)
                aug_real_i = cv2.flip(aug_real_i, -1)
            else:
                aug_comp_i = cv2.flip(aug_comp_i, 0)
                aug_mask_i = cv2.flip(aug_mask_i, 0)
                aug_real_i = cv2.flip(aug_real_i, 0)

            cv2.imwrite(save_comp_aug_path, aug_comp_i)
            cv2.imwrite(save_mask_aug_path, aug_mask_i)
            cv2.imwrite(save_real_aug_path, aug_real_i)

        if os.path.exists(save_comp_path):
            continue

        # image read and resize
        img_comp = cv2.imread(os.path.join(path, comp_path))
        img_comp = cv2.resize(img_comp, dsize = save_size, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(save_comp_path, img_comp)

        img_mask = cv2.imread(os.path.join(path, mask_path))
        img_mask = cv2.resize(img_mask, dsize = save_size, interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(save_mask_path, img_mask)

        img_real = cv2.imread(os.path.join(path, real_path))
        img_real = cv2.resize(img_real, dsize = save_size, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(save_real_path, img_real)

        f1.write('\n'+comp_aug_path)

    f1.close()
    f2.close()