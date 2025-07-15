#!/usr/bin/env python3
# Heavily based on depthai-python's "rgb_depth_aligned.py" example

# Import
import cv2
import numpy as np
import depthai as dai
import argparse
import os

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
args = parser.parse_args()
alpha = args.alpha

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

'''
# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None

    for i in range (4):
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["disp"] = None

        queueEvents = device.getQueueEvents(("rgb", "disp"))
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

        # Blend when both received
        if frameRgb is not None and frameDisp is not None:
            filenameRGB = f"rgb_image_{i}.png"
            filenameDepth=f"depth_image_{i}.png"
            cv2.imwrite(filenameRGB, frameRgb)
            print("Saved RGB image")
            cv2.imwrite(filenameDepth, frameDisp)
            print("Saved depth map")
            frameRgb = None
            frameDisp = None
            '''