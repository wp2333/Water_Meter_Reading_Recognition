# =========================================
# Time: 2020-11-17
# Author: wangping
# Function: Digit Recognition based on cnn
# =========================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from myloader import MyDataset, MyDataset_test
import csv


BATCH_SIZE = 100        
EPOCHS = 1
Lr = 0.005

transform_data = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),     # [0,1]
    ])

dataset_train = MyDataset(transform = transform_data)
# assert dataset_train
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

dataset_test = MyDataset_test(transform = transform_data)
assert dataset_test
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


class Detector(nn.Module):
   def __init__(self):
        super(Detector, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, 10)
    
    
   def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = Detector()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = Lr)


def train(epoch):
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data

        outputs = model(inputs)
        print(data,outputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))

# def generate_result(model):
#     model.eval()
#     correct = 0
#     total = 0
#     f = open('submission.csv','w',encoding='utf-8',newline=" ")
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(["filename","result"])
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, dim = 1)
            
#             csv_writer.writerow(["file_name",predicted.item()])
#             correct += (predicted == labels).sum().item()
#     f.close()

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        train(epoch)
        test()
    
    torch.save(model.state_dict(), "model/cnn.pkl")