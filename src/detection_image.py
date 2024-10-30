import sys
import cv2
import numpy as np
from openvino.runtime import Core

# OpenVINO 모델 로드
ie = Core()
model_path = "C:\\Users\\tonyp\\dev\\openvino\\fallen_people\\intel\\human-pose-estimation-0001\\FP32\\human-pose-estimation-0001.xml"
net = ie.read_model(model=model_path)
compiled_model = ie.compile_model(net, device_name="CPU")

# 입력 및 출력 레이어 설정
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


def preprocess_image(image_path, input_shape) :
    image = cv2.imread(image_path)
    # 프레임 전처리
    input_img = cv2.resize(image, (456, 256))  # Resize to (width, height)
    input_img = input_img.transpose((2, 0, 1))  # Change shape from (H, W, C) to (C, H, W)
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension (1, C, H, W)
    input_img = input_img.astype(np.float32)  # Ensure the type matches the model's input
    return (input_img, image)


def is_person_fallen(pose_points):
    # 간단한 규칙: 엉덩이와 어깨의 Y좌표가 수평에 가깝다면 쓰러짐으로 간주
    try:
        left_shoulder, right_shoulder = pose_points[1], pose_points[2]
        left_hip, right_hip = pose_points[8], pose_points[11]
        return abs(left_shoulder[1] - left_hip[1]) < 50 and abs(right_shoulder[1] - right_hip[1]) < 50
    except:
        return False


def get_bounding_box(pose_points):
    """포즈 키포인트를 이용해 바운딩 박스 좌표 계산."""
    x_coords = [point[0] for point in pose_points]
    y_coords = [point[1] for point in pose_points]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return (x_min, y_min, x_max, y_max)


def detect(image_path) :
    # image 전처리
    (input_image, image) = preprocess_image(image_path, (456, 256))
    
    # 모델 추론
    results = compiled_model([input_image])[output_layer]

    # 결과 분석 (관절 포인트 추출)
    keypoints = results.reshape(-1, 3)[:, :2]

    # 바운딩 박스 계산
    (x_min, y_min, x_max, y_max) = get_bounding_box(keypoints)


    # 쓰러진 사람 감지
    if is_person_fallen(keypoints):
        cv2.putText(image, "Person Fallen!", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 프레임 출력
    while True :
        cv2.imshow("Fall Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

detect("image2.png")