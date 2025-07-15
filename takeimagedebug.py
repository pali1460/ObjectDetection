import cv2
import numpy as np
import depthai as dai
import argparse
import os
import time

os.makedirs("images", exist_ok=True)

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
args = parser.parse_args()
alpha = args.alpha

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight

fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.Camera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
disparityOut.setStreamName("disp")
queueNames.append("disp")

#Properties
rgbCamSocket = dai.CameraBoardSocket.CAM_A

camRgb.setBoardSocket(rgbCamSocket)
camRgb.setSize(1280, 720)
camRgb.setFps(fps)

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(rgbCamSocket)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except Exception as e: # Catch specific exception or general Exception
    print(f"Could not set manual focus: {e}. Continuing without it.")
    pass # Continue without setting manual focus if it fails

left.setResolution(monoResolution)
left.setCamera("left")
left.setFps(fps)
right.setResolution(monoResolution)
right.setCamera("right")
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(rgbCamSocket)
stereo.setExtendedDisparity(False) # Try setting this to True if objects are very close
stereo.setSubpixel(False) # Try setting this to True for smoother disparities, but output will be float
# Optional: Set confidence threshold if you want to filter out low-confidence points
# stereo.setConfidenceThreshold(200) # Values from 0 to 255. Lower means more confident (and potentially fewer) points.

# Linking
camRgb.video.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)

camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    camRgb.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    # Get the output queues (blocking queues)
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
    dispQueue = device.getOutputQueue(name="disp", maxSize=4, blocking=True)

    time.sleep(2) # Give the camera some time to warm up

    for i in range(4):
        # Wait for RGB and disparity frames
        rgbPacket = rgbQueue.get()  # Blocking call
        dispPacket = dispQueue.get()  # Blocking call

        frameRgb = rgbPacket.getCvFrame()
        frameDisp = dispPacket.getFrame() # This is a UINT8 or UINT16 numpy array

        # --- Debugging and Visualization Improvements ---

        # 1. Get max disparity from the stereo node's config
        # This is the theoretical max, not necessarily the max in your scene
        maxDisparity = stereo.initialConfig.getMaxDisparity()

        print(f"Frame {i}:")
        print(f"  Disparity frame shape: {frameDisp.shape}, dtype: {frameDisp.dtype}")
        print(f"  Theoretical max disparity: {maxDisparity}")

        # Remove zero values which usually represent invalid depth (background, occlusions)
        # This helps in normalizing the *actual* valid data range
        valid_disparities = frameDisp[frameDisp != 0]

        if valid_disparities.size > 0:
            actual_min_disp = np.min(valid_disparities)
            actual_max_disp = np.max(valid_disparities)
            print(f"  Actual valid disparity range: [{actual_min_disp}, {actual_max_disp}]")

            # Option A: Normalize based on the *actual* observed range
            # This will make the best use of the colormap for the current scene
            frameDispNormalized = cv2.normalize(frameDisp, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            # Invert colors: 0 (closer) -> 255 (farther) for better visualization with COLORMAP_HOT
            # If you want closer to be brighter, set alpha=255, beta=0. If farther brighter, set alpha=0, beta=255.
            # For COLORMAP_HOT, typically closer (higher disparity) should be hotter (brighter).
            # So, normalize 0-maxDisparity to 0-255, then apply colormap.
            # If frameDisp is 0 for far, and higher for close:
            #   cv2.normalize(frameDisp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # Then apply colormap.
        else:
            print("  No valid (non-zero) disparities found in this frame.")
            frameDispNormalized = np.zeros_like(frameDisp, dtype=np.uint8)


        # Apply colormap
        # It's crucial that frameDispNormalized is 8-bit unsigned (0-255)
        # COLORMAP_HOT maps lower values to dark colors and higher values to hot/bright colors.
        # If your disparity values are such that higher values mean *closer* objects,
        # then directly mapping 0-255 to the colormap will make closer objects appear "hotter".
        frameDispColor = cv2.applyColorMap(frameDispNormalized, cv2.COLORMAP_HOT)
        frameDispColor = np.ascontiguousarray(frameDispColor) # Ensure it's contiguous for saving/displaying

        # Save images
        filenameRGB = f"images/rgb_image_{i}.png"
        filenameDisparityHeatmap = f"images/disparity_heatmap_{i}.png"
        cv2.imwrite(filenameRGB, frameRgb)
        print(f"Saved {filenameRGB}")
        cv2.imwrite(filenameDisparityHeatmap, frameDispColor)
        print(f"Saved {filenameDisparityHeatmap}")

        # Display for real-time debugging (optional, can remove for headless operation)
        # cv2.imshow(f"RGB Frame {i}", frameRgb)
        # cv2.imshow(f"Disparity Heatmap {i}", frameDispColor)
        # cv2.waitKey(1000) # Display for 1 second

    cv2.destroyAllWindows()