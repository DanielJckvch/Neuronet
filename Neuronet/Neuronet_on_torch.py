
import torch.nn as nn
import torchvision as tv
import torch as tch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

num_epochs = 5 
num_classes = 10 
batch_size = 100 
learning_rate = 0.001
DATA_PATH = 'D:\\UsersAndyPycharmProjectsMNISTData'
MODEL_STORE_PATH = 'D:\\UsersAndyPycharmProjectspytorch_models\\'

trans = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))]) 
train_dataset = tv.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True) 
test_dataset = tv.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

m = tch.Tensor([8,6,3,8,5])

class ConvNet(nn.Module): 
     def __init__(self): 
         super(ConvNet, self).__init__() 
         self.layer1 = nn.Sequential( nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
         self.layer2 = nn.Sequential( nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 
         self.drop_out = nn.Dropout() 
         self.fc1 = nn.Linear(7 * 7 * 64, 1000) 
         self.fc2 = nn.Linear(1000, 10)
     def forward(self, x): 
         out = self.layer1(x) 
         out = self.layer2(out) 
         out = out.reshape(out.size(0), -1) 
         out = self.drop_out(out) 
         out = self.fc1(out) 
         out = self.fc2(out) 
         return out

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = tch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    o = 0
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

