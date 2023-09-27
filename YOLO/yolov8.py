import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s-pose.pt')

# Open the video file
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

# 創建一個有17個關節點欄位和box欄位的DataFrame
columns = ["keypoint_" + str(i+1) for i in range(17)] + ["box"]
df = pd.DataFrame(columns=columns)

index = 1

os.makedirs("images", exist_ok=True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        boxes = results[0].boxes  # Extract bounding boxes
        # print(box.xyxy) # returns x1, y1, x2, y2

        keypoints = results[0].keypoints # Extract keypoints
        # print(keypoints.xy)  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
        # print(keypoints.xyn) # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
        # print(keypoints.conf)  # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
        # printkeypoints.data)  # raw keypoints tensor, (num_dets, num_kpts, 2/3)

        df = pd.concat([df, pd.DataFrame({"keypoint": keypoints.xy.tolist(), "box": boxes.xyxy.tolist()})], ignore_index=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Save the annotated frame
        cv2.imwrite(filename="images/" + str(index) + ".png", img=annotated_frame)
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
