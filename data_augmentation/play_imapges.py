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

# Convert Images
# Current list: BLACKWHITE, GAUSSIANBLUR, BLACKPATCHES, WHITEPATCHES, 
# RAISEBRIGHTNESS, LOWERBRIGHTNESS

def convert_images(camera_images1, camera_images2, camera_images3, augmentationType):
    match arg:
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
            # Current list: BLACKWHITE, GAUSSIANBLUR, BLACKPATCHES, WHITEPATCHES, RAISEBRIGHTNESS, LOWERBRIGHTNESS
            for arg in sys.argv[1:]:
                
                # Modify camera images
                camera_images1_mod, camera_images2_mod, camera_images3_mod = convert_images(camera_images1, camera_images2, camera_images3, arg)

                # Get new file name
                filenameMod = fileNameRoot + "_" + arg + ".hdf5"

                shutil.copy(dataset_path, filenameMod)
                with h5py.File(filenameMod, "a") as copyOfFile:
                    del copyOfFile['/observations/images']
                    copyOfFile.create_dataset("/observations/images/wrist", data=camera_images1_mod)
                    copyOfFile.create_dataset("/observations/images/ext1", data=camera_images2_mod)
                    copyOfFile.create_dataset("/observations/images/ext2", data=camera_images3_mod)





'''

dataset_path = "/home/nataliya/Downloads/luxonis_test_cup_data/data/episode_20250812_154002.hdf5"

with h5py.File(dataset_path, 'r') as root:
    actions = root['/action'][()]
    tm = root['/tm'][()]
    camera_images1 = root['/observations/images/wrist'][()]
    camera_images2 = root['/observations/images/ext1'][()]
    camera_images3 = root['/observations/images/ext2'][()]
    qpos = root['/observations/qpos'][()]



    # Scan all args and modify
    # TODO
    # Current list: BLACKWHITE, GAUSSIANBLUR, BLACKPATCHES, WHITEPATCHES, RAISEBRIGHTNESS, LOWERBRIGHTNESS
    for arg in sys.argv:


        camera_images1_gray = convert_to_gray(camera_images1)
        camera_images2_gray = convert_to_gray(camera_images2)
        camera_images3_gray = convert_to_gray(camera_images3)

        shutil.copy(dataset_path, "e1_with_grayscale.hdf5")
        with h5py.File("e1_with_grayscale.hdf5", "a") as copyOfFile:
            del copyOfFile['/observations/images']
            copyOfFile.create_dataset("/observations/images/wrist", data=camera_images1_gray)
            copyOfFile.create_dataset("/observations/images/ext1", data=camera_images2_gray)
            copyOfFile.create_dataset("/observations/images/ext2", data=camera_images3_gray)



    print(f"Episode contains {len(actions)} actions")
    print("Press 'q' during image display to skip to robot execution")

    # Display images
    for idx, img in enumerate(camera_images1):
        images = np.hstack((img, camera_images2[idx], camera_images3[idx]))
        color_image = np.asanyarray(images)
        processed_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Camera Images", color_image)
        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# TODO:
# Modify in multiple ways
# Do this to multiple files
# Make it user-friendly: Specify which augmentations to choose and take in a folder
# Take in a folder of files
# User can specify which filters to apply to them
# Augs: Black and White, Randomly black out patches of image, Gaussian blur, raise brightness, lower brightness
# Save as FILENAME_AUGMENTATION, e.g. episode_20250812_15400_BLACKWHITE or episode_20250812_15400_BLACKPATCHES
'''