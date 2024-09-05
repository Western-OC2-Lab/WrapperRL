# WrapperRL: Reinforcement Learning Agent for Feature Selection in High-Dimensional Industrial Data

This code provides the implementation of the _WrapperRL_ framework that emulates the wrapper-based method for forward feature selection using an actor-critic Reinforcement Learning (RL)
algorithm. This method is applied to image-based high-dimensional time-frequency domain data collected in an industrial environment. The main goal of _WrapperRL_ is to identify the set of features
or image patches, representing a set of frequency bands, that are affected by ambient industrial noise. In addition to the implementation details of _WrapperRL_, the code includes the implementation of the
SOTA approach _PatchDrop_ and the ablation study. 

#### The code includes the following folders: 
- `src`: is the directory that includes the _WrapperRL_ code.
- `dataset`: is the directory that includes the dataset used for building the models and obtaining the paper's results (**img_dataset.csv**).
- `models`: the directory that includes the classification model termed `SA-CNN.pt`.
- `notebooks`: the directory that includes the code for training the `SA-CNN` model. 

#### To run the corresponding codes, the following steps need to be applied:
- The **image dataset** can be downloaded using the open-source link https://drive.google.com/file/d/1h4MgVdPRQxdc8pV51ZiANcnu_M3MFThl/view?usp=sharing. Please download the dataset and put it into the `generated_images` folder.
- Alter the `IMGS_DIR` variable in `src/CNN_model/constants.py` to the absolute path of the `generated_images` directory. An example would be `r"D:\WrapperRL_AAAI\datasets\generated_images\log_all_features"`.

#### Code structure Explanation: 
The codes in (1) `src/main_grid.py`, (2) `src/main_patchDrop.py`, and (3) `src/main_randomPatch.py` follow the same structure in terms of executing a forward feature selection process to our use case. 
- The code in (1) runs the _WrapperRL_ process. In this file, the `trial_nb` variable represents the trial number, `w` represents the `lambda` variable of the reward function, and `total_episodes` represents the training epochs. During the training process, an `insights` folder is created with the corresponding `trial_nb` and `w` to store the generated models, images, statistics, and training and testing datasets for a fair assessment.
- The code in (2) runs the _PatchDrop_ process. In a similar manner, an `insights` folder is created to gather information about the process.
- The code in (3) runs the ablation study.

The codes in (1) `src/analyze_results.py` and (2) `src/analyze_patchDrop.py` analyzes the obtained results. 

# Contact-Info

Please feel free to contact me for any questions or research opportunities. 
- Email: shaeribrahim@gmail.com
- Gihub: https://github.com/ibrahimshaer and https://github.com/Western-OC2-Lab
- LinkedIn: [Ibrahim Shaer](https://www.linkedin.com/in/ibrahim-shaer-714781124/)
- Google Scholar: [Ibrahim Shaer](https://scholar.google.com/citations?user=78fAJ_IAAAAJ&hl=en) and [OC2 Lab](https://scholar.google.com/citations?user=ICvnj9EAAAAJ&hl=en)


 





  

