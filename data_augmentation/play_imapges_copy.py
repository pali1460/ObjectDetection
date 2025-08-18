import numpy as np
import cv2
import h5py
import time
import shutil


dataset_path = "e1_with_grayscale.hdf5"

dataset_path = "/home/nataliya/ObjectDetection/data_augmentation/episode_20250812_154133_RAISEBRIGHTNESS.hdf5"

with h5py.File(dataset_path, 'r') as root:
    actions = root['/action'][()]
    tm = root['/tm'][()]
    camera_images1 = root['/observations/images/wrist'][()]
    camera_images2 = root['/observations/images/ext1'][()]
    camera_images3 = root['/observations/images/ext2'][()]
    qpos = root['/observations/qpos'][()]

    print(f"Episode contains {len(actions)} actions")
    print("Press 'q' during image display to skip to robot execution")

    # Display images
    for idx, img in enumerate(camera_images1):
        images = np.hstack((img, camera_images2[idx], camera_images3[idx]))
        color_image = np.asanyarray(images)
        cv2.imshow("Camera Images", color_image)
        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()