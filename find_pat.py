import cv2
import numpy as np

def grab_cut(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)

    # GrabCut 적용
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)  # 조정된 사각형 영역
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 확실한 배경, 확실한 전경을 설정합니다.
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # GrabCut 결과에 따라 이미지를 분리합니다.
    result_image = image * mask2[:, :, np.newaxis]

    return result_image, mask2

def mark_wounds(image, template_paths):
    # SIFT 초기화
    sift = cv2.SIFT_create()

    # 결과 이미지 생성
    result_image = image.copy()

    for template_path in template_paths:
        # 이미지에서 템플릿 불러오기
        template = cv2.imread(template_path, 0)  # 그레이스케일로 템플릿 불러오기

        # 이미지에서 특징점 및 디스크립터 찾기
        kp_template, des_template = sift.detectAndCompute(template, None)
        kp_image, des_image = sift.detectAndCompute(image, None)

        # BFMatcher 객체 생성 및 매칭 수행
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_template, des_image, k=2)

        # 좋은 매칭 포인트 선택
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # 임계값 조정
                good_matches.append(m)

        # 좋은 매칭 포인트로만 특징점 좌표 추출
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 환부 주변에 사각형 그리기
        for pt in src_pts:
            x, y = pt[0]
            cv2.rectangle(result_image, (int(x - 50), int(y - 50)), (int(x + 50), int(y + 50)), (0, 0, 255), 2)  # 사각형 크기 조정

    return result_image

def adjust_brightness_and_saturation(image, brightness=0.8, saturation=0.8):
    # 이미지를 HSV 색상 모델로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 명도와 채도 조정
    hsv_image[..., 2] = np.clip(hsv_image[..., 2] * brightness, 0, 255).astype(np.uint8)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation, 0, 255).astype(np.uint8)

    # HSV에서 다시 BGR로 변환
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return adjusted_image

# 이미지 경로 설정
patient_image_path = 'patient_image.png'
template_paths = ['img1.png', 'img2.png', 'img3.png', 'img4.png']

# GrabCut을 이용하여 환부 부분 추출
extracted_wound, mask = grab_cut(patient_image_path)

# 이미지 경로 설정
patient_image_path = 'patient_image.png'

# 이미지 불러오기
image = cv2.imread(patient_image_path)

# 채도와 명도 조정
adjusted_image = adjust_brightness_and_saturation(image, brightness=0.8, saturation=0.8)

# 환부 부분에 사각형으로 체크
result_image = mark_wounds(adjusted_image, template_paths)

# 결과 이미지 출력
cv2.imshow('Marked Wounds', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
