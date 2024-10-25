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


def compute_y_distance(pt1, pt2):
    return abs(pt1[1] - pt2[1])


def draw_movement(img, point1, point2, distance):
    img_color = img.copy()
    color = (0, 255, 0)  # Color for the movement path

    cv2.circle(img_color, tuple(point2), 5, color, -1)
    cv2.line(img_color, tuple(point1), tuple(point2), color, 2)
    cv2.putText(img_color, f"Y Dist: {distance:.2f}", (point2[0] + 10, point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_color


def main(img1_path, img2_path):
    # Load YOLO model (absolute paths)
    net = cv2.dnn.readNet("/Users/suyeon/PycharmProjects/pythonProject3/20240610/yolov3.weights",
                          "/Users/suyeon/PycharmProjects/pythonProject3/20240610/yolov3.cfg")
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Detect persons
    _, person_centers1, confidences1 = detect_person_yolo(img1, net, layer_names)
    _, person_centers2, _ = detect_person_yolo(img2, net, layer_names)

    # Select the most confident detected object in the first image
    if len(person_centers1) == 0:
        print("No objects were detected in the first image.")
        return

    best_index = np.argmax(confidences1)
    selected_center1 = person_centers1[best_index]

    # Find the closest center in the second image
    if len(person_centers2) == 0:
        print("No objects were detected in the second image.")
        return

    selected_center2 = min(person_centers2, key=lambda c: compute_y_distance(selected_center1, c))

    # Compute y-axis distance
    distance = compute_y_distance(selected_center1, selected_center2)
    print(f"Object moved distance along y-axis: {distance:.2f} units")

    # Draw movement path and save result image
    img2_color = draw_movement(img2, selected_center1, selected_center2, distance)
    cv2.imwrite("YoloModel_result.png", img2_color)
    print("Result image saved as YoloModel_result.png.")


if __name__ == "__main__":
    main("1111.png", "2222.png")
