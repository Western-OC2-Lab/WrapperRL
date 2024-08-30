from torchvision import transforms


train_val_transforms = transforms.Compose(
    [
        transforms.CenterCrop(72),
        transforms.ToTensor(),
        transforms.Normalize([
            0.0004, 0.00041, 0.00041
        ], [
           0.0003, 0.00037, 0.00039
        ])
    ]
)

generated_image_transform = transforms.Compose([
    transforms.Normalize([
            0.0004, 0.00041, 0.00041
        ], [
           0.0003, 0.00037, 0.00039
        ])
])

reverse_image_transform = transforms.Normalize(mean = [
            -0.0004/0.0003, -0.00041/0.00037, -0.00041/0.00039
        ], std = [
           1/0.0003, 1/0.00037, 1/0.00039
        ])


MODEL_DIR = "../models"
MODEL_NAME = "SA-CNN.pt"
DATASET_DIR = "../datasets"
DATASET_NAME = "img_dataset.csv"
#TODO: Change the directory here!!!
# IMGS_DIR = r"D:\WrapperRL_AAAI\datasets\generated_images\log_all_features"
IMGS_DIR= ""