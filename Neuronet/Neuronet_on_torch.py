
import torch.nn as nn
import torchvision as tv
import torch as tch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from PIL import Image, ImageGrab
import numpy as np
import matplotlib.pyplot as plt

haveModel = 1
num_epochs = 30 
num_classes = 10 
batch_size = 100 
learning_rate = 0.001
TRAIN_DATA_PATH = 'D:\\Программы\\Нейросеть к учебной практике\\CIFAR-10 dataset'
MODEL_STORE_PATH = 'D:\\Программы\\Нейросеть к учебной практике\\Model\\'

class ConvNet(nn.Module): 
     def __init__(self): 
         super(ConvNet, self).__init__() 
         self.layer1 = nn.Sequential(nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
         self.layer2 = nn.Sequential(nn.Conv2d(30, 60, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
         self.drop_out = nn.Dropout() 
         self.fc1 = nn.Linear(8 * 8 * 60, 1000) 
         self.fc2 = nn.Linear(1000, 500)
         self.fc3 = nn.Linear(500, 100)
         self.fc4 = nn.Linear(100, 10)
     def forward(self, x): 
         out = self.layer1(x) 
         out = self.layer2(out) 
         out = out.reshape(out.size(0), -1) 
         out = self.drop_out(out) 
         out = self.fc1(out) 
         out = self.fc2(out) 
         out = self.fc3(out) 
         out = self.fc4(out) 
         return out


#Подготовка данных

#Тестовые данные
test_image = Image.open('image.jpg')
test_image = test_image.resize((32,32))
test_image = np.asarray(test_image, dtype='uint8')
test_image = np.transpose(test_image, (2,0,1))
#Перевод в тензор
test_image = tch.Tensor(test_image)
test_image /= 255
test_image = test_image.reshape(1,3,32,32)
test_image -= 0.1307
test_image /= 0.3081

test_label = tch.Tensor([3])

#Создание итератора данных
test_image = TensorDataset(test_image,test_label)
test_loader = DataLoader(test_image, batch_size=1, num_workers=0, shuffle=False)

model = ConvNet()
if (haveModel==0):
#У нас нет сети. Создадим её.
#Тренировочные данные
    trans = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))]) 
    train_dataset = tv.datasets.CIFAR10(root=TRAIN_DATA_PATH, train=True, transform=trans, download=True) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True) 
    #Определение функции потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = tch.optim.Adam(model.parameters(), lr=learning_rate)
    #Тренировка сети
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        
        for i, (images, labels) in enumerate(train_loader):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = tch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))
    tch.save(model.state_dict(),MODEL_STORE_PATH+'nn_model.pth')
else:
    #У нас уже есть обученная сеть
    model.load_state_dict(tch.load(MODEL_STORE_PATH+'nn_model.pth'))
    model.eval()
    #Определение класса изображения
    for i, (images, labels) in enumerate(test_loader):
        #images *=0.3081
        #images +=0.1307
        #images = images.numpy()
        #plt.imshow(np.transpose(images[0], (1, 2, 0)), cmap=plt.get_cmap('gray'))
        #plt.show()
        outputs = model(images)
        _, predicted = tch.max(outputs.data, 1)
        break
    if (predicted == labels):
        print('There is a cat on the picture')
    else:
        print('There is no cat on the picture')
#'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
