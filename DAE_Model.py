import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


class AnomalyDataset(Dataset):
    """MVTec Anomaly Detection Dataset을 PyTorch Dataset으로 구현."""
    def __init__(self, image_paths, labels, target_size=(256, 256)):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.target_size)
        img = np.array(img) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
        return img, torch.tensor(label, dtype=torch.long)


def load_images_from_folder(folder_path):
    """특정 폴더에서 모든 PNG 이미지 경로와 레이블을 반환."""
    image_paths = []
    labels = []

    for defect_type in os.listdir(folder_path):
        defect_path = os.path.join(folder_path, defect_type)
        if os.path.isdir(defect_path):
            for root, _, files in os.walk(defect_path):
                for file in files:
                    if file.lower().endswith('.png'):
                        image_paths.append(os.path.join(root, file))
                        labels.append(0 if defect_type.lower() == 'good' else 1)

    return image_paths, labels


def prepare_data(base_dir, folder_names):
    """전체 데이터셋에서 이미지 경로와 라벨을 수집."""
    all_image_paths = []
    all_labels = []

    for folder in folder_names:
        for subfolder in ['train', 'test', 'ground_truth']:
            folder_path = os.path.join(base_dir, folder, subfolder)
            image_paths, labels = load_images_from_folder(folder_path)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)

    return all_image_paths, all_labels


# CNN 기반 DAE 모델 정의
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 학습 함수 (진행률 바 추가)
def train_dae(model, dataloader, num_epochs=10, save_path="D:\MvTec\dae_model.pth"):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        with tqdm(total=len(dataloader), desc="Batch Progress") as pbar:
            for imgs, _ in dataloader:
                # 노이즈 추가
                noisy_imgs = imgs + 0.2 * torch.randn_like(imgs)
                noisy_imgs = torch.clip(noisy_imgs, 0., 1.).to(device)
                imgs = imgs.to(device)

                # 모델 학습
                optimizer.zero_grad()
                outputs = model(noisy_imgs)
                loss = criterion(outputs, imgs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)

        print(f"Epoch Loss: {epoch_loss / len(dataloader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"모델이 '{save_path}'에 저장되었습니다.")


# 결과 시각화 함수
def show_results(model, dataloader, num_images=5):
    model.eval()
    imgs, _ = next(iter(dataloader))
    noisy_imgs = imgs + 0.2 * torch.randn_like(imgs)
    noisy_imgs = torch.clip(noisy_imgs, 0., 1.).to(device)
    imgs = imgs.to(device)

    with torch.no_grad():
        outputs = model(noisy_imgs)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(imgs[i].cpu().permute(1, 2, 0))
        plt.axis("off")

        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_imgs[i].cpu().permute(1, 2, 0))
        plt.axis("off")

        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].cpu().permute(1, 2, 0))
        plt.axis("off")

    plt.show()


# 양불 판정 함수
def classify_anomalies(model, dataloader, threshold):
    model.eval()
    criterion = nn.MSELoss(reduction='mean')
    total = len(dataloader.dataset)
    correct = 0

    print("\n양불 판정 결과:")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)

            # 노이즈가 제거된 복원 이미지 예측
            outputs = model(imgs)

            # 복원 오차 계산
            mse = criterion(outputs, imgs).item()

            # 오차가 임계값보다 크면 불량(1), 작으면 양품(0)으로 분류
            prediction = 1 if mse > threshold else 0
            actual = labels.item()

            # 예측이 실제 레이블과 일치하는 경우
            if prediction == actual:
                correct += 1

            # 결과 출력
            print(f"실제 라벨: {'불량' if actual == 1 else '양품'}, 예측 라벨: {'불량' if prediction == 1 else '양품'}, MSE: {mse:.4f}")

    accuracy = correct / total * 100
    print(f"\n전체 정확도: {accuracy:.2f}%")


# 메인 실행 코드
if __name__ == "__main__":
    base_dir = r'/Users/suyeon/Downloads/mvtec_anomaly_detection'
    folder_names = ["Bottle", "Cable", "Capsule", "Carpet", "Grid", "Hazelnut", "Leather", "Metal_Nut", "Pill", "Screw",
                    "Tile", "Toothbrush"]

    # 데이터 로드 및 전처리
    image_paths, labels = prepare_data(base_dir, folder_names)
    dataset = AnomalyDataset(image_paths, labels, target_size=(256, 256))

    # 데이터셋 분할
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 배치 크기 1로 설정하여 개별 예측 확인

    # 모델 초기화 및 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    num_epochs = 2
    model_save_path = "D:\MvTec\dae_model.pth"
    train_dae(model, train_loader, num_epochs, save_path=model_save_path)

    # 저장된 모델 로드
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.to(device)
    print("저장된 모델을 불러왔습니다.")

    # 양불 판정
    classify_anomalies(model, test_loader, threshold=0.01)
