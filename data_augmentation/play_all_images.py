import numpy as np
import cv2
import h5py
import time
import shutil
import sys
import os

folder_aug = ""

if(len(sys.argv) > 1):
    folder_aug = sys.argv[1]

folder_path_beginning = "/home/nataliya/Downloads/luxonis_test_cup_data/data"

folder_path = folder_path_beginning + "/" + folder_aug


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

            print("Playing " + filename)
            print(f"Episode contains {len(actions)} actions")
            print("Press 'q' during image display to skip")

            # Display images
            for idx, img in enumerate(camera_images1):
                images = np.hstack((img, camera_images2[idx], camera_images3[idx]))
                color_image = np.asanyarray(images)
                cv2.imshow("Camera Images", color_image)
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()