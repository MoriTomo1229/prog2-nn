import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms.v2 as T

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    data_root: str = "data"
    batch_size: int = 128
    epochs: int = 280
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    log_dir: str = "runs/fashion_watch"
    ckpt_dir: str = "checkpoints"
    model: str = "cnn"  # "mlp" or "cnn"
    seed: int = 42


# -------------------------
# Models
# -------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),             # (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14
            nn.Conv2d(32, 64, 3, padding=1), # 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -------------------------
# Utils
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")  # sumで集計して後で平均
        pred = logits.argmax(dim=1)

        total_loss += loss.item()
        total_correct += (pred == y).sum().item()
        total_samples += y.numel()

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Data
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),  # 0..1
    ])

    ds_train = datasets.FashionMNIST(root=cfg.data_root, train=True, download=True, transform=transform)
    ds_test  = datasets.FashionMNIST(root=cfg.data_root, train=False, download=True, transform=transform)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    dl_test  = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    # Model
    if cfg.model == "mlp":
        model = MLP()
    else:
        model = SmallCNN()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Logging
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    global_step = 0

    # Initial eval (before training)
    test_loss, test_acc = evaluate(model, dl_test, device)
    print(f"[init] test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%")
    writer.add_scalar("test/loss", test_loss, global_step)
    writer.add_scalar("test/acc", test_acc, global_step)

    # Train loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # エポック平均（ちゃんと平均を取る）
        sum_loss = 0.0
        sum_correct = 0
        sum_samples = 0

        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{cfg.epochs}", leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            # stats
            bs = y.numel()
            sum_loss += loss.item() * bs
            sum_correct += (logits.argmax(dim=1) == y).sum().item()
            sum_samples += bs

            train_loss = sum_loss / sum_samples
            train_acc = sum_correct / sum_samples

            # tqdm表示を更新（「学習してる感」が一番出るところ）
            pbar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc*100:.2f}%")

            # tensorboard（バッチ単位）
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # epoch end eval
        test_loss, test_acc = evaluate(model, dl_test, device)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
              f"| test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%")

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/train_acc", train_acc, epoch)
        writer.add_scalar("epoch/test_loss", test_loss, epoch)
        writer.add_scalar("epoch/test_acc", test_acc, epoch)

        # checkpoint保存（あとでpredictに使える）
        ckpt_path = os.path.join(cfg.ckpt_dir, f"fashion_{cfg.model}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": cfg.__dict__}, ckpt_path)

    writer.close()
    print("done.")


if __name__ == "__main__":
    main()
