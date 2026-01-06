import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


def test_accuracy(model, dataloader):
    # 全てのミニバッチに対して推論をし、正解率を計算する
    n_corrects = 0  # 正解の個数
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            # バッチをmodelと同じデバイスに転送する
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            # モデルに入れて結果logitsをだす
            logits_batch = model(image_batch)
            predict_batch = logits_batch.argmax(dim=1)
            n_corrects += (predict_batch == label_batch).sum().item()

    accuracy = n_corrects / len(dataloader.dataset)
    return accuracy


def train(model, dataloader, loss_fn, optimizer):
    """1 epochの学習を行う"""
    # モデルのデバイスを調べる
    device = next(model.parameters()).device
    model.train()
    last_loss = None
    for image_batch, label_batch in dataloader:
        # バッチをmodelと同じデバイスに転送する
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        logits_batch = model(image_batch)
        loss = loss_fn(logits_batch, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def test(model, dataloader, loss_fn):
    """1エポック分のロスを計算"""
    model.eval()
    loss_total = 0.0
    # モデルのデバイスを調べる
    device = next(model.parameters()).device
    with torch.no_grad():
        for image_batch, label_batch in dataloader:
            # バッチをmodelと同じデバイスに転送する
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            logits_batch = model(image_batch)
            loss = loss_fn(logits_batch, label_batch)
            loss_total += loss.item()

    return loss_total / len(dataloader)
