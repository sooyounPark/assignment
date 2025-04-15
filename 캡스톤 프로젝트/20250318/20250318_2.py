import cv2

# 비디오 파일 읽기
video = cv2.VideoCapture('test_video.mp4')

# 비디오가 제대로 열렸는지 확인
if not video.isOpened():
    print("비디오 파일을 열 수 없습니다.")
else:
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # 프레임 출력
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 창 닫기
    video.release()
    cv2.destroyAllWindows()