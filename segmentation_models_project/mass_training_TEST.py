# import huggingface_hub
# huggingface_hub.accept_access_request
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from segmentation_models_pytorch.utils.metrics import IoU, Accuracy, Fscore, Precision, Recall 
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.losses import DiceLoss, BCELoss, JaccardLoss

from torch.utils.data import DataLoader
import pandas as pd
from segmentation_models_pytorch.base import SegmentationModel
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc         # garbage collect library
from dataset import Dataset #get_preprocessing
torch.cuda.is_available()
selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("TRAINING ON",selected_device)

if selected_device == "cpu":
    raise Exception("No GPU available, using CPU.")



def log_training(df:dict, train_logs: dict, validation_logs:dict):
    train_logs = {"train_"+k:v for k,v in train_logs.items()}
    validation_logs = {"val_"+k:v for k,v in validation_logs.items()}
    
    for k,v in train_logs.items():
        df[k].append(v)
    for k,v in validation_logs.items():
        df[k].append(v)
    return df

def create_log_dict(metrics, loss):
    df_train = {"train_"+metric_name.__name__:[] for metric_name in metrics}
    df_train["train_loss"] = [] # Loss is only named loss. Its type can be found in the filename
    # df_train["train_"+loss.__name__] = []
    df_val = {"val_"+metric_name.__name__:[] for metric_name in metrics}
    df_val["val_loss"] = []
    # df_val["val_"+loss.__name__] = []
    df_train.update(df_val)
    return df_train


# -------- Dataset --------
dataset_feature = "two_channel"
DATA_DIR = "../dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/compiled/compiled_datasets_16bit_large_even_labels/normal/"

x_test_dir = os.path.join(DATA_DIR, 'test/images')
y_test_dir = os.path.join(DATA_DIR, 'test/labels')

CONTINOUE_FROM_LAST = False
CHANNELS_IN_IMAGE = 2
gradient_dir_optional = "None"
if CHANNELS_IN_IMAGE == 2:
    gradient_dir_optional = "../dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/compiled/compiled_datasets_16bit_large_even_labels/gradient/"

model = torch.load("D:\Master\MasterProject\segmentation_models_project/best_model__DeepLabV3Plus__timm-efficientnet-b2__JaccardLoss__Freeze_encoder_False__two_channel__LR_0.001.pth")
model.cuda()
model.eval()

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    classes=['granule'],
    CHANNELS_IN_IMAGE=CHANNELS_IN_IMAGE,
    gradient_dir_optional=os.path.join(gradient_dir_optional, 'val/images')
)

test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False, num_workers=0, drop_last=True)

metrics = [
    IoU(),
    Accuracy(),
    Fscore(),
    Precision(), 
    Recall() 
]

loss_func = JaccardLoss() 

test_epoch = ValidEpoch(
    model, 
    loss=loss_func, 
    metrics=metrics, 
    device=selected_device,
    verbose=True,
)

# Create logging dict
df = create_log_dict(metrics, loss_func)

# Logging
test_logs = test_epoch.run(test_loader)
test_logs['loss'] = test_logs.pop(loss_func.__name__)
df = log_training(df, test_logs, test_logs) # Save stats

df = pd.DataFrame(df)
df.to_csv(f"test_results.csv")



# # ---------------- TESTING ----------------

# # load best saved checkpoint
# best_model = torch.load('./best_model.pth')

# # create test dataset
# test_dataset = Dataset(
#     x_test_dir, 
#     y_test_dir, 
#     # augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )

# test_dataloader = DataLoader(test_dataset)

# # test dataset without transformations for image visualization
# test_dataset_vis = Dataset(
#     x_test_dir, y_test_dir, 
#     classes=CLASSES,
# )


# for i in range(2):
#     n = np.random.choice(len(test_dataset))
    
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
    
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
#     visualize(
#         image=image_vis, 
#         ground_truth_mask=gt_mask, 
#         predicted_mask=pr_mask
#     )










