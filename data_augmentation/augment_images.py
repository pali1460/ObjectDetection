import numpy as np
import cv2
import h5py
import time
import shutil
import random
import sys
import os
from pathlib import Path

# Variables for configuring black/white patches

patchnum = 5
minsize = 40
maxsize = 100

# Variables for brightening/darkening

brightness_increase = 100
brightness_decrease = -50

cropHeight = 300
cropWidth = 300

def convert_to_gray(images):
    return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])

def randomly_blackout_patches(image, num_patches, patch_size_min, patch_size_max, white):
    height, width, _ = image.shape

    for _ in range(num_patches):
        patch_size = random.randint(patch_size_min, patch_size_max)

    x1 = random.randint(0, width-patch_size)
    y1 = random.randint(0, height-patch_size)
    x2 = x1+patch_size
    y2 = y1 + patch_size

    if(white):
        image[y1:y2, x1:x2] = (255, 255, 255)
    else:
        image[y1:y2, x1:x2] = (0, 0, 0)

    return image

def get_crop_coords(image, cropHeight, cropWidth):
    height, width, _ = image.shape
    if(cropHeight > height):
        raise ValueError("Patch height exceeds image height!")
    if(cropWidth > width):
        raise ValueError("Patch width exceeds image width!")
    
    cropX = random.randint(0, width-cropWidth)
    cropY = random.randint(0, height-cropHeight)
    return cropX, cropY

def crop_patch(image, cropHeight, cropWidth, cropX, cropY):
    return image[cropY:cropY + cropHeight, cropX:cropX+cropWidth]

def randomly_crop_patch(image, cropHeight, cropWidth):
    cropX, cropY = get_crop_coords(image, cropHeight, cropWidth)
    return image[cropY:cropY + cropHeight, cropX:cropX+cropWidth]

def black_patches(images):
    return np.array([randomly_blackout_patches(img, patchnum, minsize, maxsize, False) for img in images])

def white_patches(images):
    return np.array([randomly_blackout_patches(img, patchnum, minsize, maxsize, True) for img in images])

def gaussian_blur(images):
    return np.array([cv2.GaussianBlur(img, (5,5), 0) for img in images])

def raise_brightness(images):
    return np.array([cv2.convertScaleAbs(img, alpha=1.0, beta=brightness_increase) for img in images])

def lower_brightness(images):
    return np.array([cv2.convertScaleAbs(img, alpha=1.0, beta=brightness_decrease) for img in images])

def random_crop(images):
    cropX, cropY = get_crop_coords(images[0], cropHeight, cropWidth)
    return np.array([crop_patch(img, cropHeight, cropWidth, cropX, cropY) for img in images])

def random_random_crop(images):
    return np.array([randomly_crop_patch(img, cropHeight, cropWidth) for img in images])

# Convert Images
# Current list: BLACKWHITE, GAUSSIANBLUR, BLACKPATCHES, WHITEPATCHES, 
# RAISEBRIGHTNESS, LOWERBRIGHTNESS, RANDOMCROP, RANDOMRANDOMCROP

def convert_images(camera_images1, camera_images2, camera_images3, augmentationType):
    match augmentationType:
        case "BLACKWHITE":
            return convert_to_gray(camera_images1), convert_to_gray(camera_images2), convert_to_gray(camera_images3)
        case "GAUSSIANBLUR":
            return gaussian_blur(camera_images1), gaussian_blur(camera_images2), gaussian_blur(camera_images3)
        case "BLACKPATCHES":
            return black_patches(camera_images1), black_patches(camera_images2), black_patches(camera_images3)
        case "WHITEPATCHES":
            return white_patches(camera_images1), white_patches(camera_images2), white_patches(camera_images3)
        case "RAISEBRIGHTNESS":
            return raise_brightness(camera_images1), raise_brightness(camera_images2), raise_brightness(camera_images3)
        case "LOWERBRIGHTNESS":
            return lower_brightness(camera_images1), lower_brightness(camera_images2), lower_brightness(camera_images3)
        case "RANDOMCROP":
            return random_crop(camera_images1), random_crop(camera_images2), random_crop(camera_images3)
        case "RANDOMRANDOMCROP":
            return random_random_crop(camera_images1), random_random_crop(camera_images2), random_random_crop(camera_images3)
        case _:
            raise ValueError("Invalid augmentation type")

# Cycle over all HDF5 files in the fodler

folder_path = '/home/nataliya/Downloads/luxonis_test_cup_data/data'

for filename in os.listdir(folder_path):

    # Only scan .hdf5
    if filename.endswith(('.hdf5')):

        # Get dataset path
        dataset_path = os.path.join(folder_path, filename)

        with h5py.File(dataset_path, 'r') as root:

            actions = root['/action'][()]
            tm = root['/tm'][()]
            camera_images1 = root['/observations/images/wrist'][()]
            camera_images2 = root['/observations/images/ext1'][()]
            camera_images3 = root['/observations/images/ext2'][()]
            qpos = root['/observations/qpos'][()]

            # Get file name for second part
            fileNameRoot = Path(dataset_path).stem

            # Scan all args and modify
            # See above for valid args
            for arg in sys.argv[1:]:
                print("Applying augmentation " + arg + " to " + filename)
                
                # Modify camera images
                camera_images1_mod, camera_images2_mod, camera_images3_mod = convert_images(camera_images1, camera_images2, camera_images3, arg)

                # Get new file name
                filenameMod = fileNameRoot + "_" + arg + ".hdf5"

                finalFile = folder_path + "/augmented/" + filenameMod

                shutil.copy(dataset_path, finalFile)
                with h5py.File(finalFile, "a") as copyOfFile:
                    del copyOfFile['/observations/images']
                    copyOfFile.create_dataset("/observations/images/wrist", data=camera_images1_mod)
                    copyOfFile.create_dataset("/observations/images/ext1", data=camera_images2_mod)
                    copyOfFile.create_dataset("/observations/images/ext2", data=camera_images3_mod)