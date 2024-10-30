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

def detect(image_path) :
    image = cv2.imread(image_path)

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

    # Display the output frame
    while  True :
        cv2.imshow("Multiple Person Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    sys.exit(0)


detect("pedestrian.png")


