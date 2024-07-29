import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt

# ハイパーパラメータ
sequence_length = 30  # 各シーケンスのフレーム数（ここでは各フレームを1シーケンスとするため1）
num_keypoints = 17 * 2  # 17個のキーポイント、それぞれにxとyの座標
num_classes = 2  # SafeとDanger
batch_size = 32
learning_rate = 0.001
num_epochs = 20
hidden_size = 128
num_layers = 4
kernel_size = 3

# データセットの定義
class PoseDataset(Dataset):
    def __init__(self, keypoints_dir, max_keypoints):
        keypoints_files = glob.glob(os.path.join(keypoints_dir, '*.csv'))

        self.keypoints_df = pd.concat((pd.read_csv(f) for f in keypoints_files), ignore_index=True)

        # キーポイントデータの整形
        self.data = self.keypoints_df.groupby('Frame').apply(
            lambda df: self.pad_or_trim(df.iloc[:, 3:].values.flatten(), max_keypoints)).values

        # 仮のラベルを作成 (すべてのフレームに対して0)
        self.labels = np.zeros(len(self.data), dtype=int)

        # デバッグ: データのサイズを確認
        print(f'Data size: {self.data.shape}')
        print(f'Labels size: {self.labels.shape}')

    def pad_or_trim(self, keypoints, max_keypoints):
        # パディングまたはトリミングを行い、指定した長さに合わせる
        if len(keypoints) > max_keypoints:
            return keypoints[:max_keypoints]
        else:
            return np.pad(keypoints, (0, max_keypoints - len(keypoints)), 'constant')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # データを (seq_len, num_keypoints) の形状に変換
        keypoints = self.data[idx].reshape(-1, num_keypoints)
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# モデルの構築
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
        if x.dim() == 2:  # 2次元テンソルの場合
            x = x.unsqueeze(0)  # バッチサイズが1の場合、次元を追加する
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len) に変換
        x = self.tcn(x)
        x = x[:, :, -1]  # 最後のタイムステップの出力を使用
        x = self.fc(x)
        return x

# モデル、損失関数、オプティマイザの定義
input_size = num_keypoints

model = LongConvActionRecognitionModel(input_size, hidden_size, num_layers, num_classes, kernel_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# データローダーの準備
train_keypoints_dir = '/home/umelab3d/workspace/researchlab-project/third/ViTPose-Pytorch/new_keypoint_csv'
val_keypoints_dir = '/home/umelab3d/workspace/researchlab-project/third/ViTPose-Pytorch/new_val_keypoint_csv'

max_keypoints = num_keypoints * sequence_length

train_dataset = PoseDataset(train_keypoints_dir, max_keypoints)
val_dataset = PoseDataset(val_keypoints_dir, max_keypoints)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 結果を保存するパス
output_json_path = '/home/umelab3d/workspace/researchlab-project/third/ViTPose-Pytorch/training_results.json'
model_path = '/home/umelab3d/workspace/researchlab-project/third/ViTPose-Pytorch/pose_action_recognition_model.pth'

# トレーニングループとバリデーションループ
training_results = []

# グラフ描画用リスト
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    training_results.append({
        'epoch': epoch + 1,
        'loss': epoch_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # エポックごとのグラフを表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy over Epochs')

    plt.pause(1)  # 1秒間表示
    plt.close()

# 結果をJSONファイルに保存
with open(output_json_path, 'w') as f:
    json.dump(training_results, f, indent=4)

# モデルを保存
torch.save(model.state_dict(), model_path)

# 最終的なグラフを表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy over Epochs')

plt.show()

print("Training complete and model saved.")
