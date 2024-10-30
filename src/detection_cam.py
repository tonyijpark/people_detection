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

# 비디오 스트림 열기 (웹캠 또는 파일)
cap = cv2.VideoCapture(0)  # 0: 웹캠 사용, 파일 사용 시 "video.mp4"

def is_person_fallen(pose_points):
    # 간단한 규칙: 엉덩이와 어깨의 Y좌표가 수평에 가깝다면 쓰러짐으로 간주
    try:
        left_shoulder, right_shoulder = pose_points[1], pose_points[2]
        left_hip, right_hip = pose_points[8], pose_points[11]
        return abs(left_shoulder[1] - left_hip[1]) < 50 and abs(right_shoulder[1] - right_hip[1]) < 50
    except:
        return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리
    input_img = cv2.resize(frame, (456, 256))  # Resize to (width, height)
    input_img = input_img.transpose((2, 0, 1))  # Change shape from (H, W, C) to (C, H, W)
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension (1, C, H, W)
    input_img = input_img.astype(np.float32)  # Ensure the type matches the model's input
    
    # 모델 추론
    results = compiled_model([input_img])[output_layer]

    # 결과 분석 (관절 포인트 추출)
    keypoints = results.reshape(-1, 3)[:, :2]

    # 쓰러진 사람 감지
    if is_person_fallen(keypoints):
        cv2.putText(frame, "Person Fallen!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 프레임 출력
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()