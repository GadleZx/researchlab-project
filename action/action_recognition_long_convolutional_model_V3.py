import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ディレクトリの指定
train_directory = '/home/umelab3d/workspace/researchlab-project/third/ViTPose-Pytorch/new_keypoint_csv'
val_directory = '/home/umelab3d/workspace/researchlab-project/third/ViTPose-Pytorch/new_val_keypoint_csv'

# Hyperparameters
sequence_length = 30  # number of frames in each sequence
num_keypoints = 17 * 2  # 17 keypoints with x, y coordinates
num_classes = 2  # Safe and Danger
batch_size = 32
learning_rate = 0.001
num_epochs = 20
hidden_size = 128
num_layers = 4
kernel_size = 3
validation_split = 0.2

# モデルの定義
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LongConvActionRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, kernel_size):
        super(LongConvActionRecognitionModel, self).__init__()
        num_channels = [hidden_size] * num_layers
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len) for TCN
        x = self.tcn(x)
        x = x[:, :, -1]  # use the output of the last time step
        x = self.fc(x)
        return x

# データセットの定義
class PoseDataset(Dataset):
    def __init__(self, keypoints, labels, tracking_ids):
        self.keypoints = torch.tensor(keypoints, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tracking_ids = torch.tensor(tracking_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.keypoints[idx], self.labels[idx], self.tracking_ids[idx]

# CSVファイルをディレクトリから読み込む関数
def load_csv_files_from_directory(directory):
    all_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            all_files.append(df)
    return pd.concat(all_files, ignore_index=True)

# トレーニングとバリデーションデータの読み込み
df_train = load_csv_files_from_directory(train_directory)
df_val = load_csv_files_from_directory(val_directory)

# 必要なデータの抽出
X_train = df_train.drop(columns=['Label', 'Tracking_ID']).values
y_train = df_train['Label'].values
tracking_ids_train = df_train['Tracking_ID'].values

X_val = df_val.drop(columns=['Label', 'Tracking_ID']).values
y_val = df_val['Label'].values
tracking_ids_val = df_val['Tracking_ID'].values

# データの形状を (num_samples, sequence_length, num_keypoints) に変換
def reshape_data(X, y, tracking_ids, sequence_length, num_keypoints):
    num_samples = len(X) // (sequence_length * num_keypoints)
    X = X[:num_samples * sequence_length * num_keypoints].reshape(num_samples, sequence_length, num_keypoints)
    y = y[:num_samples * sequence_length:sequence_length]
    tracking_ids = tracking_ids[:num_samples * sequence_length:sequence_length]
    return X, y, tracking_ids

X_train, y_train, tracking_ids_train = reshape_data(X_train, y_train, tracking_ids_train, sequence_length, num_keypoints)
X_val, y_val, tracking_ids_val = reshape_data(X_val, y_val, tracking_ids_val, sequence_length, num_keypoints)

# データセットとデータローダーの作成
train_dataset = PoseDataset(X_train, y_train, tracking_ids_train)
val_dataset = PoseDataset(X_val, y_val, tracking_ids_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# モデル、損失関数、最適化関数の定義
input_size = num_keypoints
model = LongConvActionRecognitionModel(input_size, hidden_size, num_layers, num_classes, kernel_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習過程の損失と精度を記録するリスト
train_losses = []
val_losses = []
val_accuracies = []

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, labels, tracking_ids in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # バリデーション
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels, tracking_ids in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {epoch_loss:.4f}, '
          f'Validation Loss: {val_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}')

print("Training complete.")

# 損失と精度のグラフをプロットする関数
def plot_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 損失のグラフ
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 精度のグラフ
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# グラフを表示
plot_metrics(train_losses, val_losses, val_accuracies)


# バリデーションデータセット内の各CSVファイルの精度を計算する関数
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels, tracking_ids in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 各ファイルの精度を計算
file_accuracies = []

val_files = [f for f in os.listdir(val_directory) if f.endswith('.csv')]
for filename in val_files:
    df = pd.read_csv(os.path.join(val_directory, filename))
    X_val = df.drop(columns=['Label', 'Tracking_ID']).values
    y_val = df['Label'].values
    tracking_ids_val = df['Tracking_ID'].values

    # データの形状を (num_samples, sequence_length, num_keypoints) に変換
    X_val, y_val, tracking_ids_val = reshape_data(X_val, y_val, tracking_ids_val, sequence_length, num_keypoints)

    val_dataset = PoseDataset(X_val, y_val, tracking_ids_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    accuracy = calculate_accuracy(model, val_loader)
    file_accuracies.append((filename, accuracy))
    print(f'File: {filename}, Validation Accuracy: {accuracy:.4f}')

# グラフで精度を比較
file_names = [name for name, _ in file_accuracies]
accuracies = [acc for _, acc in file_accuracies]

plt.figure(figsize=(12, 6))
plt.bar(file_names, accuracies, color='skyblue')
plt.xlabel('CSV Files')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy for Each CSV File')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

