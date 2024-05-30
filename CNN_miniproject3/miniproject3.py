import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 이미지 읽기 함수 정의
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):  # 이미지 파일 확장자를 확인합니다.
            label = int(filename.split('_')[0])
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # 흑백 이미지로 변환
            img = img.resize((256, 256))  # 이미지 크기를 256x256으로 조정
            img = np.array(img)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# 학습 및 테스트 이미지 로드
train_images, train_labels = load_images_from_folder('../CNN_miniproject3/mnt/data/hangul_dataset/train')
test_images, test_labels = load_images_from_folder('../CNN_miniproject3/mnt/data/hangul_dataset/test')

# 이미지 데이터의 형태 변경 및 정규화
train_images = train_images.reshape((train_images.shape[0], 256, 256, 1)).astype("float32") / 255
test_images = test_images.reshape((test_images.shape[0], 256, 256, 1)).astype("float32") / 255

# 2. CNN 분류기 모델링
inputs = keras.Input(shape=(256, 256, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
cnn_cls = keras.Model(inputs=inputs, outputs=outputs)

cnn_cls.summary()

cnn_cls.compile(optimizer="rmsprop",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

# 3. Convolution Neural Network 분류기 학습 및 성능평가
history = cnn_cls.fit(train_images, train_labels, epochs=10, batch_size=64)
test_loss, test_acc = cnn_cls.evaluate(test_images, test_labels)
print(f"테스트 정확도: {test_acc:.3f}")
print(f"테스트 Loss: {test_loss:.3f}")

# 학습 정확도 그래프 그리기
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# 학습 손실 그래프 그리기
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()