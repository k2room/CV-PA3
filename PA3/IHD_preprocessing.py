import os 

path = '/home/work/CV-PA3/PA3/dataset_processing/'
with open(os.path.join(path, 'IHD_train.txt'), 'r') as f:
    train_list = f.readlines()
    train_list = [i.strip() for i in train_list]

with open(os.path.join(path, 'IHD_train_aug.txt'), 'w') as ff: # HAdobe5k/composite_images/a5000_1_5.jpg
    for i in train_list:
        part = i.split('/')
        line = part[0]+'_aug/'+part[1]+'/'+part[2]+'\n'
        ff.write(line)