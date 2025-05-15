import cv2
import mediapipe as mp
import json
import os
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

mp_to_op_mapping = {
    0: 0,   
    2: 1, 
    5: 2,  
    7: 3,   
    8: 4,  
    11: 5,  
    12: 6,  
    13: 7, 
    14: 8, 
    15: 9, 
    16: 10, 
    23: 11, 
    24: 12, 
    25: 13, 
    26: 14, 
    27: 15, 
    28: 16  
}

image_dir = r"C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch\assets\image"
output_dir = r"C:\Users\Home\Downloads\Virtual_try_on_skratch\Virtual_try_on_skratch\assets\pose"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png')):
        image_path = os.path.join(image_dir, img_file)
        image = cv2.imread(image_path)

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        openpose_keypoints = [[0, 0, 0] for _ in range(18)]  

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            for mp_idx, op_idx in mp_to_op_mapping.items():
                if mp_idx < len(keypoints):
                    x = keypoints[mp_idx].x
                    y = keypoints[mp_idx].y
                    openpose_keypoints[op_idx] = [x, y, 1.0]

            if 11 in mp_to_op_mapping and 12 in mp_to_op_mapping:
                left_shoulder = keypoints[11]
                right_shoulder = keypoints[12]
                neck = [(left_shoulder.x + right_shoulder.x) / 2,
                        (left_shoulder.y + right_shoulder.y) / 2,
                        1.0]
                openpose_keypoints[17] = neck 

        flat_keypoints = [coord for kp in openpose_keypoints for coord in kp]

        openpose_output = {
            "version": 1.3,
            "people": [
                {"pose_keypoints_2d": flat_keypoints}
            ]
        }

        base_name = os.path.splitext(img_file)[0]
        json_name = f"{base_name}_keypoints.json"
        json_path = os.path.join(output_dir, json_name)

        with open(json_path, 'w') as f:
            json.dump(openpose_output, f)

print("✅ MediaPipe keypoints converted and saved in OpenPose format at:", output_dir)
