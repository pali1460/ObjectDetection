# Import
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
leftOut = pipeline.create(dai.node.XLinkOut)   # Added for left camera output
rightOut = pipeline.create(dai.node.XLinkOut)  # Added for right camera output

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
disparityOut.setStreamName("disp")
queueNames.append("disp")
leftOut.setStreamName("left")  # Set stream name for left camera
queueNames.append("left")
rightOut.setStreamName("right") # Set stream name for right camera
queueNames.append("right")


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
except:
    raise

left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
left.setResolution(monoResolution)
left.setFps(fps)
right.setResolution(monoResolution)
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(rgbCamSocket)

# Linking
camRgb.video.link(rgbOut.input)
stereo.syncedLeft.link(leftOut.input)
stereo.syncedRight.link(rightOut.input)

stereo.disparity.link(disparityOut.input)
left.out.link(leftOut.input)   # Link left camera output
right.out.link(rightOut.input) # Link right camera output

camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    camRgb.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)

savedDepth = False
savedRGB = False
savedLeft = False  # Flag for left image
savedRight = False # Flag for right image


# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None
    frameLeft = None   # Variable to store left frame
    frameRight = None  # Variable to store right frame

    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["disp"] = None
        latestPacket["left"] = None # Initialize for left frame
        latestPacket["right"] = None # Initialize for right frame


        queueEvents = device.getQueueEvents(tuple(queueNames)) # Use all queue names
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()

        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
            frameDisp = np.ascontiguousarray(frameDisp)

        if latestPacket["left"] is not None: # Get left frame
            frameLeft = latestPacket["left"].getCvFrame()
            # Mono frames are typically grayscale, convert to BGR for consistency if needed
            if len(frameLeft.shape) < 3:
                frameLeft = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2BGR)

        if latestPacket["right"] is not None: # Get right frame
            frameRight = latestPacket["right"].getCvFrame()
            # Mono frames are typically grayscale, convert to BGR for consistency if needed
            if len(frameRight.shape) < 3:
                frameRight = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2BGR)

        # Wait for depth frames to load (and other frames)
        time.sleep(2)

        # Blend when both received
        if frameRgb is not None and frameDisp is not None:
            # Need to have both frames in BGR format before blending
            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)

        # Saving depth image
        if frameDisp is not None and not savedDepth:
            filenameDisparityHeatmap = f"images/depth_image.png"
            cv2.imwrite(filenameDisparityHeatmap, frameDisp)
            print(f"Saved {filenameDisparityHeatmap}")
            savedDepth = True

        # Saving RGB image
        if frameRgb is not None and not savedRGB:
            filenameRGB = f"images/rgb_image.png"
            cv2.imwrite(filenameRGB, frameRgb)
            print(f"Saved {filenameRGB}")
            savedRGB = True

        # Saving left camera image
        if frameLeft is not None and not savedLeft:
            filenameLeft = f"images/left_image.png"
            cv2.imwrite(filenameLeft, frameLeft)
            print(f"Saved {filenameLeft}")
            savedLeft = True

        # Saving right camera image
        if frameRight is not None and not savedRight:
            filenameRight = f"images/right_image.png"
            cv2.imwrite(filenameRight, frameRight)
            print(f"Saved {filenameRight}")
            savedRight = True

        if savedRGB and savedDepth and savedLeft and savedRight:
            print("All images saved. Exiting.")
            break