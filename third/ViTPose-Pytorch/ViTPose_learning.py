import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseDataset(Dataset):
    def __init__(self, keypoints_file, labels_file):
        self.keypoints_data = pd.read_csv(keypoints_file)
        self.labels_data = pd.read_csv(labels_file)

        # ラベルデータをフレーム番号でマージ
        self.data = pd.merge(self.keypoints_data, self.labels_data, on='Frame')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 骨格点データの平坦化
        keypoints = self.flatten_keypoints(self.data.iloc[idx, :])
        label = self.data.iloc[idx, -1]
        label = 1 if label == 'Danger' else 0
        return {'keypoints': keypoints, 'label': label}

    @staticmethod
    def flatten_keypoints(row):
        keypoints = []
        for i in row[:-1]:
            if isinstance(i, str):
                # タプルとして解釈される文字列を実際のタプルに変換
                kp = eval(i)
                keypoints.extend(kp[:2])  # x, y 座標のみを使用
        return torch.tensor(keypoints, dtype=torch.float32)


class PoseCNN(nn.Module):
    def __init__(self, input_size):
        super(PoseCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * ((input_size - 2) // 2 - 2), 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # チャネルの次元を追加
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, criterion, optimizer, train_loader, epochs=5):
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs = data['keypoints']
            labels = data['label']

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data['keypoints']
            labels = data['label']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test data: {100 * correct / total}%')

def main():
    keypoints_file = 'keypoint.csv'
    labels_file = 'frame_label_data_002_slip.csv'

    dataset = PoseDataset(keypoints_file, labels_file)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=4)  # 仮に同じデータセットを使用

    input_size = len(dataset[0]['keypoints'])
    model = PoseCNN(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer, train_loader, epochs=5)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
