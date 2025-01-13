import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

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

"""print("PyTorch에서 지원하는 CUDA 버전:", torch.version.cuda)

if torch.cuda.is_available():
    print("CUDA is available! GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

print("CPU 코어 수 (os.cpu_count):", os.cpu_count())"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)

# 학습 함수 (진행률 바 추가)
def train_dae(model, dataloader, num_epochs=10, save_path="D:\MvTec\dae_model.pth"):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scaler = torch.amp.GradScaler('cuda')  # 혼합 정밀도 학습을 위한 스케일러
    best_loss = float('inf')

    # 손실 및 정확도 기록을 위한 리스트
    epoch_losses = []
    val_losses = []
    batch_losses = []
    train_accuracies = []
    val_accuracies = []
    val_mse_hist = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1} Progress", leave=True) as pbar:
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                noisy_imgs = imgs + 0.2 * torch.randn_like(imgs)
                noisy_imgs = torch.clamp(noisy_imgs, 0., 1.).to(device)

                optimizer.zero_grad()

                # 혼합 정밀도 학습 적용
                with torch.amp.autocast('cuda'):
                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, imgs)

                # 역전파
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batch_losses.append(loss.item())  # 배치 손실 기록

                # 정확도 계산을 위한 처리 (복원된 이미지와 원본 이미지의 차이가 적은지 비교)
                mse = nn.functional.mse_loss(outputs, imgs, reduction='none').mean([1, 2, 3])
                predictions = (mse < 0.02).long()  # 임의의 임계값(예: 0.02)으로 양/불을 결정
                correct_train += (predictions == labels).sum().item()
                total_train += labels.size(0)

                # tqdm에 표시할 정보 업데이트
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        average_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(average_epoch_loss)  # 에포크 손실 기록
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)  # 학습 정확도 기록

        print(f"Epoch Loss: {average_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation 손실 및 정확도 계산
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        mse_list = []
        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)

                outputs = model(val_imgs)
                loss = criterion(outputs, val_imgs)
                val_loss += loss.item()

                # 정확도 계산
                mse = nn.functional.mse_loss(outputs, val_imgs, reduction='none').mean([1, 2, 3])
                mse_list.extend(mse.cpu().numpy())
                predictions = (mse < 0.02).long()
                correct_val += (predictions == val_labels).sum().item()
                total_val += val_labels.size(0)

        average_val_loss = val_loss / len(val_loader)
        val_losses.append(average_val_loss)  # 검증 손실 기록
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)  # 검증 정확도 기록
        val_mse_hist.extend(mse_list)  # 검증 데이터셋의 MSE 기록

        log_message("학습 완료")
        print(f"Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 최적의 모델 저장
        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"새로운 최적의 모델이 '{save_path}'에 저장되었습니다. (Loss: {best_loss:.4f})")

    # 학습 종료 후 손실 그래프 저장
    plt.figure(figsize=(18, 15))

    # 1. 에포크 손실 그래프 (Training Loss per Epoch)
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid()

    # 2. 검증 손실 그래프 (Validation Loss per Epoch)
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', color='orange', label='Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.legend()
    plt.grid()

    # 3. 학습 정확도 그래프 (Training Accuracy per Epoch)
    plt.subplot(2, 3, 3)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='green',
             label='Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.legend()
    plt.grid()

    # 4. 검증 정확도 그래프 (Validation Accuracy per Epoch)
    plt.subplot(2, 3, 4)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', color='red',
             label='Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid()

    # 5. 배치 손실 그래프 (Batch-wise Training Loss)
    plt.subplot(2, 3, 5)
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, marker='.', label='Training Loss per Batch', alpha=0.6)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.grid()

    # 6. 검증 MSE 히스토그램 (Validation MSE Histogram)
    plt.subplot(2, 3, 6)
    plt.hist(val_mse_hist, bins=50, color='purple', alpha=0.7)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Validation MSE Histogram')
    plt.grid()

    plt.tight_layout()
    #plt.show()
    plt.savefig("training_results.png")
    print("손실 그래프가 'training_results.png'에 저장되었습니다.")

    # 모델 저장 (CPU로 이동 후 저장)
    save_path = "D:\MvTec\dae_model_last.pth"
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

def log_message(message):
    """시간을 포함한 로그 메시지를 출력합니다."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

# 메인 실행 코드
if __name__ == "__main__":
    base_dir = r'D:\MvTec\mvtec_anomaly_detection'
    folder_names = ["Bottle", "Cable", "Capsule", "Carpet", "Grid", "Hazelnut", "Leather", "Metal_Nut", "Pill", "Screw",
                    "Tile", "Toothbrush"]

    log_message("데이터 로드 및 전처리")
    image_paths, labels = prepare_data(base_dir, folder_names)
    dataset = AnomalyDataset(image_paths, labels, target_size=(256, 256))

    log_message("데이터셋 분할")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    log_message("DataLoader 생성")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 배치 크기 1로 설정하여 개별 예측 확인

    log_message("모델 초기화 및 학습")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    num_epochs = 200
    model_save_path = "D:\MvTec\dae_model.pth"
    train_dae(model, train_loader, num_epochs, save_path=model_save_path)

    log_message("저장된 모델 로드")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.to(device)
    print("저장된 모델을 불러왔습니다.")

    log_message("양불 판정")
    classify_anomalies(model, test_loader, threshold=0.01)
