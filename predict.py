import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


# モデルをインスタンス化する
model = models.MyModel()
print(model)

# データセットのロード
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype=torch.float32, scale=True)])
)

image, target = ds_train[0]

image = image.unsqueeze(dim=0)

model
with torch.no_grad():
    logits = model(image)

print(logits)

# plt.bar(range(len(logits[0])), logits[0])
# plt.show()

plt.subplot(1, 2, 1)
plt.imshow(image[0].squeeze(), cmap='gray_r', vmin=0, vmax=1)

# plt.figure()
probs = logits.softmax(dim=1)


plt.subplot(1, 2, 2)

plt.bar(range(len(probs[0])), probs[0])
plt.ylim(0, 1)
plt.show()
