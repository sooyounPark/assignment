import cv2
import numpy as np

def detect_person_yolo(img, net, layer_names, conf_threshold=0.5, nms_threshold=0.4):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    centers = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > conf_threshold:  # '0' class ID is for 'person' in COCO dataset
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                centers.append((center_x, center_y))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()

    person_boxes = [boxes[i] for i in indices]
    person_centers = [centers[i] for i in indices]
    person_confidences = [confidences[i] for i in indices]

    return person_boxes, person_centers, person_confidences

def match_points(points1, points2):
    matched_points2 = []
    for pt1 in points1:
        closest_pt = min(points2, key=lambda pt2: np.linalg.norm(np.array(pt1) - np.array(pt2)))
        matched_points2.append(closest_pt)
    return matched_points2

def draw_movement(img1, img2, points1, points2):
    img_color = img2.copy()
    color = (0, 255, 0)  # Color for the movement path

    for point1, point2 in zip(points1, points2):
        cv2.circle(img_color, tuple(point2), 5, color, -1)
        cv2.line(img_color, tuple(point1), tuple(point2), color, 2)
        cv2.putText(img_color, f"({point2[0]}, {point2[1]})", (point2[0] + 10, point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img_color, f"({point1[0]}, {point1[1]})", (point1[0] + 10, point1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_color

def triangulate_points(points1, points2, K1, K2, R, T):
    # Convert points to homogeneous coordinates
    points1_hom = cv2.convertPointsToHomogeneous(points1).reshape(-1, 3).T
    points2_hom = cv2.convertPointsToHomogeneous(points2).reshape(-1, 3).T

    # Projection matrices
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))

    # Triangulate points
    points_4D_hom = cv2.triangulatePoints(P1, P2, points1_hom[:2], points2_hom[:2])
    points_3D = points_4D_hom / points_4D_hom[3]

    return points_3D[:3].T

def main(img1_path, img2_path):
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Detect persons using YOLO
    _, person_centers1, _ = detect_person_yolo(img1, net, layer_names)
    _, person_centers2, _ = detect_person_yolo(img2, net, layer_names)

    # Check if detections are made
    if len(person_centers1) == 0:
        print("No objects were detected in the first image.")
        return

    if len(person_centers2) == 0:
        print("No objects were detected in the second image.")
        return

    # Match points between the two images
    matched_centers2 = match_points(person_centers1, person_centers2)

    # Example intrinsic parameters
    K1 = np.array([[1000, 0, img1.shape[1] // 2],
                   [0, 1000, img1.shape[0] // 2],
                   [0, 0, 1]], dtype=float)
    K2 = np.array([[1000, 0, img2.shape[1] // 2],
                   [0, 1000, img2.shape[0] // 2],
                   [0, 0, 1]], dtype=float)

    # Example extrinsic parameters (rotation and translation)
    R = np.eye(3)
    T = np.array([[-0.1], [0], [0]], dtype=float)

    # Triangulate points to find 3D coordinates
    points3D = triangulate_points(np.array(person_centers1), np.array(matched_centers2), K1, K2, R, T)
    print(f"3D coordinates:\n{points3D}")

    # Draw movement path and save result image
    img2_color = draw_movement(img1, img2, person_centers1, matched_centers2)
    cv2.imwrite("YoloModel3_result.png", img2_color)
    print("Result image saved as YoloModel3_result.png.")

if __name__ == "__main__":
    main("1111.png", "2222.png")