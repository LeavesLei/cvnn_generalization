from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import sys
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from keras.datasets import imdb


class Dataset():
    def __init__(self, x, y, transform=None):
        assert(len(x) == len(y))
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform( Image.fromarray(x) )
            # x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


def load_mnist(dataset='mnist', batch_size=1024):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    if dataset=='mnist':
        train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
        test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)
    elif dataset=='fashionmnist':
        train_set = datasets.FashionMNIST('../data', train=True, transform=trans, download=True)
        test_set = datasets.FashionMNIST('../data', train=False, transform=trans, download=True)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=False)
    num_classes = 10

    return testloader, trainloader, num_classes


def load_cifar(dataset='cifar10', batch_size=128):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.]),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.]),
    ])

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes



def TinyImageNet(root='./path', train=True, transform=None):
    if train:
        path = '{}/tiny-imagenet/train.npz'.format(root)
    else:
        path = '{}/tiny-imagenet/test.npz'.format(root)

    data = np.load(path)

    return Dataset(x=data['images'], y=data['labels'], transform=transform)



def load_tinyimagenet(data_dir='../data', batch_size=128):

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1., 1., 1.]),
    ])

    trainset = TinyImageNet(data_dir, train=True, transform=trans)
    testset = TinyImageNet(data_dir, train=False, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 200

    return trainloader, testloader, num_classes


def load_imdb(batch_size=512):
    nb_words = 1000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def vectorize_sequences(sequences, dimension=nb_words):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    # Convert training data to bag-of-words:
    x_train = vectorize_sequences(x_train)
    x_test = vectorize_sequences(x_test)

    # Convert labels from integers to floats:
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    trainset = Dataset(x=x_train, y=y_train, transform=None)
    testset = Dataset(x=x_test, y=y_test, transform=None)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_classes = 2
    return trainloader, testloader, num_classes