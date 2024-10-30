import cv2

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

    # 프레임 화면에 표시
    cv2.imshow('Video Player', frame)

    # 'q' 키를 누르면 재생 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()