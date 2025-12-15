import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

sd_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(dtype=torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=sd_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=sd_transform
)
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size,
)

# for image_batch, label_batch in dataloader_train:
#     print(image_batch.shape)
#     print(label_batch.shape)
#     break

model = models.MyModel()

# acc_test = models.MyModel.test_accuracy(model, dataloader_test)
# print(f'test accuracy:{acc_test*100:.3f}%')
loss_fm = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 20

train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/{n_epochs}')

    time_start = time.time()
    train_loss = models.train(model, dataloader_train, loss_fm, optimizer)
    time_end = time.time()
    print(f'        training loss:  {train_loss:.3f} ({time_end-time_start:.3f}s)')
    train_loss_log.append(train_loss)

    time_start = time.time()
    val_loss = models.test(model, dataloader_test, loss_fm)
    time_end = time.time()
    print(f'    validation loss: {val_loss:.3f} ({time_end-time_start:.3f}s)')
    val_loss_log.append(val_loss)

    time_start = time.time()
    train_acc = models.test_accuracy(model, dataloader_train)
    time_end = time.time()
    print(f'    training accuracy: {train_acc*100:.3f}% ({time_end-time_start:.3f}s)')
    train_acc_log.append(train_acc)

    time_start = time.time()
    val_acc = models.test_accuracy(model, dataloader_test)
    time_end = time.time()
    print(f'    validation accuracy: {val_acc*100:.3f}%' f' ({time_end-time_start:.3f}s)')
    val_acc_log.append(val_acc)


plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs + 1), train_loss_log)
plt.xticks(range(1, n_epochs + 1))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs + 1), val_acc_log)
plt.xticks(range(1, n_epochs + 1))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.show()

#    fig, ax = plt.subplots()
#    ax.plot(range(1, n_epochs+1), train_loss_log)
#    ax.set_xlabel('epochs')
#    ax.set_ylabel('loss')
#    ax.grid()
#    plt.show()
