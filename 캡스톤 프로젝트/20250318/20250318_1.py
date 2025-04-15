import cv2

# 이미지 읽기
image = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)

# 이미지가 제대로 읽혔는지 확인
if image is None:
    print("이미지를 읽을 수 없습니다.")
else:
    # 이미지 출력
    cv2.imshow('Lenna', image)
    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()  # 모든 창 닫기