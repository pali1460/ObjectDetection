import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Define and configure the left mono camera node
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(30)

# Create an XLinkOut node for the left camera stream
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.out.link(xoutLeft.input)

# Define and configure the right mono camera node
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)  # Set to RIGHT camera
monoRight.setFps(30)

# Create an XLinkOut node for the right camera stream
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.out.link(xoutRight.input) # Link the right camera output

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Get the output queues for both camera streams
    left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        # Get the latest frame from the left camera queue
        in_left = left_queue.tryGet()
        if in_left is not None:
            frame_left = in_left.getCvFrame()
            cv2.imshow("Left Camera Feed", frame_left)

        # Get the latest frame from the right camera queue
        in_right = right_queue.tryGet()
        if in_right is not None:
            frame_right = in_right.getCvFrame()
            cv2.imshow("Right Camera Feed", frame_right)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
