import time

import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from torch.nn import functional as F
import warnings

#测试
# 压制警告
warnings.filterwarnings('ignore')

model = torch.load(r'model_path/model_100.pth')
model.eval()
img_path = r'test_path/airplane/airplane_10.png'
image = Image.open(img_path)

image.show()

# 避免不同格式照片的问题
image = image.convert('RGB')
# 改变图片格式并改为Tensor格式
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
image = transform(image)
print('the size of image before reshape:', image.size())

image = torch.reshape(image, (1, 3, 32, 32)).cuda()

start = time.time()
out = model(image)
end = time.time()
print('用时: ', end - start)

out = F.softmax(out, dim=1)
print(out.tolist())


print()

y = out.argmax(1)
y = y.tolist()

class_list = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('类别: ', class_list[y[0]])
