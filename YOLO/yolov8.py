import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s-pose.pt')

# Open the video file
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

df = pd.DataFrame(columns = ["keypoint", "box"])

index = 1

os.makedirs("images", exist_ok = True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        boxes = results[0].boxes  # Bounding boxes of detected objects
        # print(box.xyxy) # returns x1, y1, x2, y2

        keypoints = results[0].keypoints  # Masks object
        # print(keypoints.xy)  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
        df = df.append({"keypoint": str(keypoints.xy.tolist()), "box": str(boxes.xyxy.tolist())}, ignore_index = True)
        # print(keypoints.xyn) # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
        # print(keypoints.conf)  # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
        # printkeypoints.data)  # raw keypoints tensor, (num_dets, num_kpts, 2/3)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Save the annotated frame
        cv2.imwrite(filename = "images/" + str(index) + ".png", img = annotated_frame)
        index += 1

        # Resize image
        annotated_frame = cv2.resize(annotated_frame, (960, 540))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# 寫入CSV文件
df.to_csv('output.csv', index=False)