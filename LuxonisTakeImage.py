#!/usr/bin/env python3
# Heavily based on depthai-python's "rgb_depth_aligned.py" example

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
except:
    raise
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

    time.sleep(1)

    for i in range(4):
        # Wait for RGB and disparity frames
        rgbPacket = rgbQueue.get()  # Blocking call
        dispPacket = dispQueue.get()  # Blocking call

        frameRgb = rgbPacket.getCvFrame()
        frameDisp = dispPacket.getFrame()

        maxDisparity = stereo.initialConfig.getMaxDisparity()
        frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
        frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
        frameDisp = np.ascontiguousarray(frameDisp)

        # Save images
        filenameRGB = f"images/rgb_image_{i}.png"
        filenameDepth = f"images/depth_image_{i}.png"
        cv2.imwrite(filenameRGB, frameRgb)
        print(f"Saved {filenameRGB}")
        cv2.imwrite(filenameDepth, frameDisp)
        print(f"Saved {filenameDepth}")