import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


# For MNIST and FashionMNIST
class MNIST_ComplexNet(nn.Module):
    
    def __init__(self):
        super(MNIST_ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(4*4*20, 500)
        self.fc2 = ComplexLinear(500, 10)
             
    def forward(self,x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,4*4*20)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x


class CIFAR_ComplexNet(nn.Module):
    
    def __init__(self):
        super(CIFAR_ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(3, 10, 5, 1)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(5*5*20, 500)
        self.fc2 = ComplexLinear(500, 100)
             
    def forward(self,x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,5*5*20)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x

# for tiny imagenet
class TinyImageNet_ComplexNet(nn.Module):
    
    def __init__(self):
        super(TinyImageNet_ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(3, 10, 5, 1)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.conv3 = ComplexConv2d(20, 20, 5, 1)

        self.fc1 = ComplexLinear(4*4*20, 500)
        self.fc2 = ComplexLinear(500, 200)
             
    def forward(self,x):
        x = self.conv1(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = complex_relu(x)
        x = complex_max_pool2d(x, 2, 2)
        x = x.view(-1,4*4*20)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x

# for IMDB
class IMDB_ComplexNet(nn.Module):
    
    def __init__(self):
        super(IMDB_ComplexNet, self).__init__()

        self.fc1 = ComplexLinear(1000, 500)
        self.fc2 = ComplexLinear(500, 200)
        self.fc3 = ComplexLinear(200, 2)
             
    def forward(self,x):
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = complex_relu(x)
        x = self.fc3(x)
        x = x.abs()
        x =  F.log_softmax(x, dim=1)
        return x