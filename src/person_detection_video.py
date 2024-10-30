import sys
import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO Core and load the person detection model
ie = Core()
model_path = "C:\\Users\\tonyp\\dev\\openvino\\fallen_people\\intel\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.xml"
compiled_model = ie.compile_model(model=model_path, device_name="CPU")

# Get the input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def detect(image) :

    # Prepare the input image
    height, width = image.shape[:2]
    input_image = cv2.resize(image, (544, 320))  # 모델 요구 크기
    input_image = input_image.transpose((2, 0, 1))  # HWC -> CHW
    input_image = np.expand_dims(input_image, axis=0)  # 배치 차원 추가
    input_image = input_image.astype(np.float32)

    # Perform inference
    results = compiled_model([input_image])[output_layer]

    # Process the detection results
    # Process the detection results
    for detection in results[0][0]:
        conf = detection[2]
        if conf > 0.5:  # 신뢰도가 50% 이상인 경우에만 처리
            xmin = int(detection[3] * width)
            ymin = int(detection[4] * height)
            xmax = int(detection[5] * width)
            ymax = int(detection[6] * height)

            # 감지된 사람 영역 표시
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                image, f"Person: {conf:.2f}", (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )


    return image


# 동영상 파일 경로 설정
video_path = "pedestrian.mp4"  # 재생할 동영상 파일명 또는 경로

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

# 동영상 재생 루프
while True:
    ret, frame = cap.read()  # 한 프레임씩 읽기

    if not ret:  # 읽을 프레임이 없는 경우 종료
        print("동영상 재생이 끝났습니다.")
        break


    image = detect(frame)

    # 프레임 화면에 표시
    cv2.imshow('Video Player', frame)

    # 'q' 키를 누르면 재생 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

