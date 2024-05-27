# import huggingface_hub
# huggingface_hub.accept_access_request
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU, Accuracy, Fscore
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.losses import DiceLoss, BCELoss, JaccardLoss, BCEWithLogitsLoss
from custom_losses import FocalLoss, BinaryLovaszLoss, BCEJaccardLoss, BCEDiceLoss

from torch.utils.data import DataLoader
from itertools import product
import pandas as pd
from pathlib import Path
from segmentation_models_pytorch.base import SegmentationModel
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc         # garbage collect library
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import Dataset #get_preprocessing
from plots import visualize, create_log_dict, log_training
import plotly.graph_objects as go
import time
from datetime import timedelta
torch.cuda.is_available()
selected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("TRAINING ON",selected_device)

if selected_device == "cpu":
    raise Exception("No GPU available, using CPU.")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Create folder structure
Path("./MODELS").mkdir(parents=True, exist_ok=True)
Path("./TRAINING_RESULTS").mkdir(parents=True, exist_ok=True) # tmux -> ctrl+s to see all sessions, -> ctrl+b d to detach from session

# -------- Dataset --------
# DATA_DIR = "D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/compiled_datasets_16bit_tiny"
# DATA_DIR = "D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/compiled_datasets_16bit_medium_equal_label_size/gradient"
dataset_feature = "two_channel"
DATA_DIR = "../dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/compiled_datasets_16bit_medium_equal_label_size/normal/"
x_train_dir = os.path.join(DATA_DIR, 'train/images')
y_train_dir = os.path.join(DATA_DIR, 'train/labels')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')
y_valid_dir = os.path.join(DATA_DIR, 'val/labels')

x_test_dir = os.path.join(DATA_DIR, 'test/images')
y_test_dir = os.path.join(DATA_DIR, 'test/labels')

CONTINOUE_FROM_LAST = True
CHANNELS_IN_IMAGE = 2
gradient_dir_optional = "None"
if CHANNELS_IN_IMAGE == 2:
    gradient_dir_optional = "../dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/compiled_datasets_16bit_medium_equal_label_size/gradient/"

#TODO: Implement Focal loss, use kaggle steel competition version

# models = [smp.DeepLabV3Plus, smp.MAnet, smp.PSPNet, smp.FPN, smp.PAN, smp.Linknet, smp.UnetPlusPlus] 
models = [smp.DeepLabV3Plus] 
ENCODERS = ['timm-efficientnet-b2']#, 'tu-xception41', 'tu-resnetv2_101', 'tu-resnetv2_50', 'resnet101', 'resnet34'] # 'tu-xception71' <- batch 10
# ENCODERS = ['resnet101'] #['resnet101', 'mobilenet_v2']#, 'efficientnet-b0', 'resnet34']
loss_functions = [JaccardLoss(), BinaryLovaszLoss(), BCEJaccardLoss(), BCEDiceLoss(), DiceLoss(), BCELoss(), DiceLoss(), BCELoss()] 
freeze = [False]#, True] # True 

for i, data in enumerate(product(models, ENCODERS, loss_functions, freeze)):
    # torch.cuda.reset_peak_memory_stats()

    architecture, ENCODER, loss_func, freeze_encoder = data
    print("\n ------------------------")
    print(f"Now training: {architecture()._get_name()} with {ENCODER} using {loss_func._get_name()} | Encoder freeze: {freeze_encoder} | Data feature: {dataset_feature}")

    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['granule']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = selected_device

    model: SegmentationModel = architecture(
        encoder_name=ENCODER, # Backbone 
        encoder_weights=ENCODER_WEIGHTS, # Pre-trained weights 
        classes=len(CLASSES), # Only one class, 'The granule'
        activation=ACTIVATION,
        in_channels=CHANNELS_IN_IMAGE # For grayscale
    )
    # ----- Skip training if already done -----
    potential_file = Path(f"./TRAINING_RESULTS/best_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}.csv")
    if potential_file.exists():
        if not CONTINOUE_FROM_LAST:
            print(f"Results already exists for {model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}. Skipping \n")
            continue
        model = torch.load(f"./MODELS/best_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}.pth")

    # model.load_state_dict()
    # Use same preprocessing method as the encodes trained dataset
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir, #x_test_dir, #x_train_dir, 
        y_train_dir, #y_test_dir, #y_train_dir, 
        classes=CLASSES,
        CHANNELS_IN_IMAGE=CHANNELS_IN_IMAGE,
        gradient_dir_optional=os.path.join(gradient_dir_optional, 'train/images')
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES,
        CHANNELS_IN_IMAGE=CHANNELS_IN_IMAGE,
        gradient_dir_optional=os.path.join(gradient_dir_optional, 'val/images')
    )
    if CHANNELS_IN_IMAGE == 1:
        assert (train_dataset[0][0].shape == (1,1024,1024)), f"Data was loaded wrong! Expected {CHANNELS_IN_IMAGE} channels, but got {train_dataset[0][0].shape[0]}"
        assert (train_dataset[0][1].shape == (1,1024,1024)), f"Data was loaded wrong! Expected {CHANNELS_IN_IMAGE} channels, but got {train_dataset[0][1].shape[0]}"

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False, num_workers=0, drop_last=True)

    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(z=valid_dataset[0][1].squeeze(), colorscale='Inferno'))
    # fig.update_layout(width=1200, height=1000, showlegend=False, font_size=20)
    # fig.show()
    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(z=valid_dataset[0][0].squeeze(), colorscale='Inferno'))
    # fig.update_layout(width=1200, height=1000, showlegend=False, font_size=20)
    # fig.show()
    # exit()
    metrics = [
        IoU(),
        Accuracy(),
        Fscore()
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    schedular = ReduceLROnPlateau(optimizer=optimizer, factor=0.6, mode="min", patience=1, verbose=True)


    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = TrainEpoch(
        model, 
        loss=loss_func, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model, 
        loss=loss_func, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    if freeze_encoder == True:
        for p in model.encoder.parameters():
            p.requires_grad = False


    max_score = 0
    # Create logging dict
    df = create_log_dict(metrics, loss_func)

    for i in range(0, 40): # TODO: Implement early stopping
        
        print(f"\nEpoch: {i} | {model._get_name()}__{ENCODER}__{loss_func._get_name()}")
        # Logging
        train_logs = train_epoch.run(train_loader)
        train_logs['loss'] = train_logs.pop(loss_func.__name__)
        valid_logs = valid_epoch.run(valid_loader)
        valid_logs['loss'] = valid_logs.pop(loss_func.__name__)
        df = log_training(df, train_logs, valid_logs) # Save stats
        # Step learning rate
        schedular.step(valid_logs['loss'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']: 
            max_score = valid_logs['iou_score']
            torch.save(model, f"./MODELS/best_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}.pth")
            print('Model saved!')
            
        # if i == 7:
        #     optimizer.param_groups[0]['lr'] = 1e-5
        #     print('Decrease decoder learning rate to 1e-5!')
    # Save last model
    # torch.save(model, f"./MODELS/last_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()__Freeze_encoder_{freeze_encoder}}.pth")

    # Test inference speed of best model
    best_model = torch.load(f"./MODELS/best_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}.pth")
    cycles = 10
    times = []
    for i in range(cycles):
        x = torch.from_numpy(train_dataset[i][0]).to(selected_device).unsqueeze(0)
        # start = time.time()
        starttime = time.perf_counter()
        best_model.predict(x)
        # end = time.time()
        # times.append(end - start)
        duration = time.perf_counter()-starttime
        times.append(duration)
    inference_time = np.mean(np.array(times))

    df = pd.DataFrame(df)
    # Calc # parameters in model
    decoder_total_params = sum(p.numel() for p in model.decoder.parameters())
    encoder_total_params = sum(p.numel() for p in model.encoder.parameters())
    seg_head_total_params = sum(p.numel() for p in model.segmentation_head.parameters())
    df['parameters'] = seg_head_total_params+encoder_total_params+decoder_total_params
    # Calc amount of memory used
    # df['memory_used'] = torch.cuda.max_memory_allocated()
    # df['memory_used'] = torch.cuda.max_memory_reserved()
    ## torch.cuda.reset_max_memory_allocated()
    # torch.cuda.reset_peak_memory_stats()
    # Add if encoder was freezed during training
    df['freeze_encoder'] = freeze_encoder
    # Add estimated inference time
    df['inference_time'] = inference_time
    # Save 
    if potential_file.exists() and CONTINOUE_FROM_LAST: # Merge
        exiting_df = pd.read_csv(potential_file, index_col=0)
        df = pd.concat((exiting_df, df), ignore_index=True)
        df.to_csv(f"./TRAINING_RESULTS/best_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}.csv")
    else: # Overwrite exiting file
        df.to_csv(f"./TRAINING_RESULTS/best_model__{model._get_name()}__{ENCODER}__{loss_func._get_name()}__Freeze_encoder_{freeze_encoder}__{dataset_feature}.csv")
    
    # Memory handling
    del model
    del optimizer
    del train_epoch
    del valid_epoch
    del train_loader
    del valid_loader
    del decoder_total_params
    del encoder_total_params
    del seg_head_total_params
    del schedular

    # gc.collect()
    # with torch.cuda.device(selected_device):
    #     torch.cuda.empty_cache() 

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










