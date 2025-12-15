from torchvision import io as tvio
from torchvision import models
import torchinfo

imput_image = tvio.decode_image('assets/IMG_4419.jpg')
print(type(imput_image))
print(imput_image.shape, imput_image.dtype)

weights = models.ConvNeXt_Tiny_Weights.DEFAULT

model = models.convnext_tiny(weights=weights)
print(model)

torchinfo.summary(model)

preprocess = weights.transforms()

batch = preprocess(imput_image).unsqueeze(dim=0)
print(batch.shape)

model.eval()

output_logits = model(batch)
print(output_logits.shape, output_logits.dtype)

output_probs = output_logits.softmax(dim=1)

class_id = output_probs[0].argmax().item()
score = output_probs[0][class_id].item()
category_name = weights.meta['categories'][class_id]
print(f'{category_name}: {100*score:.1f}%')
