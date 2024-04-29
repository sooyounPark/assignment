import cv2
import numpy as np


def grab_cut(image_path):
    # 이미지 로드와 기본적인 오류 처리
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

    # 초기 마스크와 모델을 생성하여 GrabCut 알고리즘 적용
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result_image = image * mask2[:, :, np.newaxis]
    return result_image


def preprocess_image(image):
    # 이미지 명암 대비 증가
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def mark_wounds(image, template_paths):
    sift = cv2.SIFT_create()
    result_image = image.copy()

    for template_path in template_paths:
        template = cv2.imread(template_path, 0)
        if template is None:
            raise FileNotFoundError(f"Template image at {template_path} could not be loaded.")

        kp_template, des_template = sift.detectAndCompute(template, None)
        kp_image, des_image = sift.detectAndCompute(image, None)

        if des_template is None or des_image is None:
            continue  # Skip if no descriptors can be computed

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_template, des_image, k=2)

        # RANSAC을 사용한 이상치 제거
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Draw bounding box in Red
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            image = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            matchesMask = None

    return image


# 이 함수들을 실행하는 코드는 여기에 위치하게 됩니다.
# 예를 들면, 위에서 정의한 함수들을 호출하는 코드 조각입니다:
try:
    patient_image_path = 'patient_image.png'
    template_paths = ['immg2-1.png', 'immg2-2.png', 'immg2-3.png', 'immg2-4.png']

    patient_image = grab_cut(patient_image_path)
    preprocessed_image = preprocess_image(patient_image)
    result_image = mark_wounds(preprocessed_image, template_paths)

    # 결과 이미지를 파일로 저장하거나 디스플레이 할 수 있습니다.
    cv2.imwrite('result_image.png', result_image)
    # cv2.imshow('Marked Wounds', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
except Exception as e:
    print(f"An error occurred: {e}")
