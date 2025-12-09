import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

ds_train = datasets.FashionMNIST(
    root='datasets',
    train=True,
    download=True
    )

print(f'datasets size: {len(ds_train)}')

image, target = ds_train[0]

print(type(image))
print(target)

# plt.imshow(image, cmap='gray_r', vmin=0, vmax=255)
# plt.title(target)
# plt.show()

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray_r', vmin=0, vmax=255)
ax.set_title(target)
plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, dtype=torch.float32, scale=True)
print(type(image))
print(image.shape, image.dtype)
print(image.min(), image.max())
