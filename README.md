# computer vision pa #3
Env set & data download : https://github.com/junleen/RainNet
```
python preprocessing.py # Image resize to 512*512
```

# result

IN2 implement -> checkpoints/experiment_IN2_train, evaluated_IN  
latest_net_G : PSNR 32.6730, MSE 117.3482  
5_net_G : PSNR 33.7992, MSE 89.6806

RAIN implement -> checkpoints/experiment_RAIN_train, evaluated_RAIN  
latest_net_G : PSNR 34.7988, MSE 66.9516  
3_net_G : 


# data augmentation
random flip and random rotate
```
python IHD_preprocessing.py # Add augmentation image path
```
```
python preprocessing.py # set augmentation = True
```

# saved model
https://drive.google.com/drive/folders/1vA8T9HoRXCDy_PAWCGTR1G-Vi2XAfpA4?usp=share_link
