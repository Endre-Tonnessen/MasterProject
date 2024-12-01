from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import numpy as np
import albumentations as albu

class Dataset(BaseDataset):
    """Granule Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['granule']
    
    def __init__(
            self,  # TODO: MAKE COMPATIBLE WITH 2-CHANNEL IMAGE BASED ON BOTH DATASETS
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            CHANNELS_IN_IMAGE=1,
            gradient_dir_optional=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.CHANNELS_IN_IMAGE = CHANNELS_IN_IMAGE

        if CHANNELS_IN_IMAGE == 2:
            self.gradient_dir_optional = os.listdir(gradient_dir_optional)
            self.gradient_dir_optional_fps = [os.path.join(gradient_dir_optional, image_id) for image_id in self.gradient_dir_optional]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        if self.CHANNELS_IN_IMAGE == 1:
            image = cv2.imread(self.images_fps[i], cv2.IMREAD_UNCHANGED)
            image = np.expand_dims(image, axis=0).astype('float32')
            assert image.shape == (1,1024,1024)
            # assert image[524,524] > 260, "Ensure 16bit"
        elif self.CHANNELS_IN_IMAGE == 2:
            normal = cv2.imread(self.images_fps[i], cv2.IMREAD_UNCHANGED)
            gradient = cv2.imread(self.gradient_dir_optional_fps[i], cv2.IMREAD_UNCHANGED) 

            normal = np.expand_dims(normal, axis=0)
            gradient = np.expand_dims(gradient, axis=0)
            
            upscaled_2channel_image = np.concatenate((normal, gradient), axis=0).astype('float32')
            # upscaled_2channel_image = np.expand_dims(upscaled_2channel_image, 0).astype('float32')
            image = upscaled_2channel_image
            # print(image.shape)
            # assert image.shape == (1,2,1024,1024)
        # elif self.CHANNELS_IN_IMAGE == 3:
        #     image = cv2.imread(self.images_fps[i])
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise Exception(f"CHANNELS_IN_IMAGE was {self.CHANNELS_IN_IMAGE}, only 1, and 2 are supported!")
        
        mask = cv2.imread(self.masks_fps[i], 0)
        # mask = np.stack([mask], axis=-1).astype('float32')
        mask = np.expand_dims(mask, 0).transpose(0,1,2).astype('float32')
        # mask = np.expand_dims(mask, 0)
        # print(type(mask))
        # print(mask.shape)
        
        # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        # if self.preprocessing:
        #     print(mask.shape)
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        #     print(mask.shape)
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
# def to_tensor(x, **kwargs):
#     if x.shape == (1024,1024): # Custom reshape of greyscale (1-channel) images
#         # x = torch.from_numpy(x.astype('float32')).unsqueeze(0).permute(0,1,2)
#         x = np.expand_dims(x, 0).transpose(0,1,2).astype('float32')
#         # print(x.shape)
#         # print(type(x))
#         return x
#     elif x.shape == (2,1024,1024): # 2-channel image
#         x = np.expand_dims(x, 0).astype('float32')
#         return x
    # Handle 3-channel
    # return x.astype('float32')
    # return x.transpose(2, 0, 1).astype('float32')

# def get_preprocessing(CHANNELS_IN_IMAGE=3):
#     """Construct preprocessing transform
    
#     Args:
#         preprocessing_fn (callbale): data normalization function 
#             (can be specific for each pretrained neural network)
#     Return:
#         transform: albumentations.Compose
    
#     """
#     return albu.Compose([albu.Lambda(image=to_tensor, mask=to_tensor)], ) # TODO: Remove
    
    # _transform = []
    # if CHANNELS_IN_IMAGE == 3: # Only apply encoder preprocess if images are 3-channel, greyscale does not support it.
    #     _transform.append(albu.Lambda(image=preprocessing_fn))

    # _transform.append(albu.Lambda(image=to_tensor, mask=to_tensor))
    # return albu.Compose(_transform)