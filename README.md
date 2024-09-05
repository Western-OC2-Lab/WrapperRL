# WrapperRL: Reinforcement Learning Agent for Feature Selection in High-Dimensional Industrial Data

This code provides the implementation of the _WrapperRL_ framework that emulates the wrapper-based method for forward feature selection using an actor-critic Reinforcement Learning (RL)
algorithm. This method is applied to image-based high-dimensional time-frequency domain data collected in an industrial environment. The main goal of _WrapperRL_ is to identify the set of features
or image patches, representing a set of frequency bands, that are affected by ambient industrial noise. In addition to the implementation details of _WrapperRL_, the code includes the implementation of the
SOTA approach _PatchDrop_ and the ablation study. 

#### The code includes the following folders: 
- `src`: is the directory that includes the _WrapperRL_ code.
- `dataset`: is the directory that includes the dataset used for building the models and obtaining the paper's results (**img_dataset.csv**).

#### To run the corresponding codes, the following steps need to be applied:
- The **image dataset** can be downloaded using the open-source link https://drive.google.com/file/d/1h4MgVdPRQxdc8pV51ZiANcnu_M3MFThl/view?usp=sharing. Please download the dataset and put it into the `generated_images` folder.
- Alter the `IMGS_DIR` variable in `src/CNN_model/constants.py` to the absolute path of the `generated_images` directory. An example would be `r"D:\WrapperRL_AAAI\datasets\generated_images\log_all_features"`.



  

