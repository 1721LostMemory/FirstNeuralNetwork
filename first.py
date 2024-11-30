import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
import torch.optim as optim
import matplotlib.pyplot as plt

isTrained = True
epochs = 1

def main():
  # 数据预处理
  transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])

  # 下载并加载训练集
  trainSet = datasets.CIFAR10(root='CIFAR10', train=True, download=True, transform=transform)
  trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)

  # 下载并加载测试集
  testSet = datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transform)
  testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False, num_workers=2)

  # 定义神经网络
  class MyFirstNN(nn.Module):
    def __init__(self):
      super(MyFirstNN, self).__init__()
      self.conv1 = nn.Conv2d(3, 6, 5) # 32 - 5 + 1 = 28
      self.pool = nn.MaxPool2d(2, 2) # 28 - 2 / 2 + 1 = 14
      self.conv2 = nn.Conv2d(6, 16, 5) # 14 - 5 + 1 = 10
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

    def forward(self, x): 
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x

  net = MyFirstNN()

  # 定义损失函数和优化器
  lossFn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用 Adam 优化器

  # 加载模型参数
  if isTrained:
    net.load_state_dict(torch.load('./netParameters.pth', weights_only=True))
    print("模型参数已加载，继续训练...")

  # 开始训练
  for epoch in range(epochs):
    runningLoss = 0.0

    for i, data in enumerate(trainLoader, 0):
      inputs, labels = data

      optimizer.zero_grad()  # 梯度清零

      outputs = net(inputs)  # 前向传播

      loss = lossFn(outputs, labels)
      loss.backward()  # 反向传播
            
      optimizer.step()  # 更新参数
            
      runningLoss += loss.item()
            
      if (i + 1) % 200 == 0:
        print(f"epoch: {epoch + 1}, batch: {i + 1}, loss: {runningLoss / 200}")
        runningLoss = 0

  print('训练完成！')

  # 开始测试并可视化结果
  dataIter = iter(testLoader)
  images, labels = next(dataIter)

  # 进行预测
  outputs = net(images)
  _, predicted = torch.max(outputs, 1)

  def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    img = np.clip(img, 0, 1)  # 确保值在 [0, 1] 范围内
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 调整维度
    plt.axis('off')

  # 显示图像和预测结果
  imshow(utils.make_grid(images))
  plt.title(f'True: {labels.numpy()}\nPred: {predicted.numpy()}')
  plt.show()

  # 保存模型  
  torch.save(net.state_dict(), './netParameters.pth')

  correct = 0
  total = 0

  with torch.no_grad():
    for data in testLoader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f"accuracy: {correct / total:.2f}")

if __name__ == '__main__':
  main()