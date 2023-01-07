import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# アノテーション
from corrupt_rabel import corrupt_label

# ニューラルネットワークモデルの定義
from model import Net

def run(criterion, annotator, true_label=False):
    # ----------------------------------------------------------
    # input:
    #   criterion:損失関数
    #   annotator:アノテータ―がランダムにラベル付けを行う確率

    # ----------------------------------------------------------
    # ハイパーパラメータなどの設定値
    num_epochs = 10         # 学習を繰り返す回数
    num_batch = 100         # 一度に処理する画像の枚数
    learning_rate = 0.001   # 学習率
    image_size = 28*28      # 画像の画素数(幅x高さ)

    # ----------------------------------------------------------
    # GPU(CUDA)が使えるかどうか？
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------------
    # 学習用／評価用のデータセットの作成
    # 変換方法の指定
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # ----------------------------------------------------------
    # MNISTデータの取得
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    # 学習用
    train_dataset = datasets.MNIST(
        root='./data',        # データの保存先
        train=True,           # 学習用データを取得する
        download=True,        # データが無い時にダウンロードする
        transform=transform   # テンソルへの変換など
    )
    # 評価用
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # ----------------------------------------------------------
    # 真のラベルを保存して、疑似アノテーション

    num_training = len(train_dataset.train_labels)

    if true_label == True:
        true_labels = np.copy(train_dataset.train_labels)

    corrupt_label(train_dataset.train_labels, annotator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=num_batch,
        shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=num_batch,
        shuffle=True)

    # ----------------------------------------------------------
    # ニューラルネットワークの生成
    model = Net(image_size, 10).to(device)

    # ----------------------------------------------------------
    # 最適化手法の設定
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------------------------------------------
    # 学習
    model.train()  # モデルを訓練モードにする

    loss_log = []

    for epoch in range(num_epochs):  # 学習を繰り返し行う
        tmp_log = []  # epoch毎のlossのログ
        loss_sum = 0

        for inputs, labels in train_dataloader:

            # GPUが使えるならGPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)

            # optimizerを初期化
            optimizer.zero_grad()

            # ニューラルネットワークの処理を行う
            inputs = inputs.view(-1, image_size)  # 画像データ部分を一次元へ並び変える
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            loss = criterion(outputs, labels)
            tmp_log.append(float(loss))
            loss_sum += loss

            # 勾配の計算
            loss.backward()

            # 重みの更新
            optimizer.step()

        # 学習状況の表示
        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")
        loss_log.append(tmp_log)

        # モデルの重みの保存
        torch.save(model.state_dict(), 'model_weights.pth')

    # ----------------------------------------------------------
    # 評価
    model.eval()  # モデルを評価モードにする

    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:

            # GPUが使えるならGPUにデータを送る
            inputs = inputs.to(device)
            labels = labels.to(device)

            # ニューラルネットワークの処理を行う
            inputs = inputs.view(-1, image_size)  # 画像データ部分を一次元へ並び変える
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            loss_sum += criterion(outputs, labels)

            # 正解の値を取得
            pred = outputs.argmax(1)
            # 正解数をカウント
            correct += pred.eq(labels.view_as(pred)).sum().item()

    print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
    