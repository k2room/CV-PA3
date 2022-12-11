# computer vision PA #3
Env set & data download : https://github.com/junleen/RainNet
```
python preprocessing.py # Image resize to 512*512
```

# result

IN implement -> checkpoints/experiment_IN2_train, evaluated_IN2  
latest_net_G, 5_net_G

RAIN implement -> checkpoints/experiment_RAIN_train, evaluated_RAIN  
latest_net_G, 3_net_G


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
