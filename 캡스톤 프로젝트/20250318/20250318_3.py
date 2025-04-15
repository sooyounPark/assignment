import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread('Lenna.png')

# 이미지가 제대로 읽혔는지 확인
if image is None:
    print("이미지를 읽을 수 없습니다.")
else:
    # 각 색상 채널 분리
    blue_channel, green_channel, red_channel = cv2.split(image)

    # 각 채널을 출력
    cv2.imshow('Blue Channel', blue_channel)
    cv2.imshow('Green Channel', green_channel)
    cv2.imshow('Red Channel', red_channel)

    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()  # 모든 창 닫기