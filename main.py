
import argparse
import seaborn as sns   
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from models import MNIST_ComplexNet, CIFAR_ComplexNet, TinyImageNet_ComplexNet, IMDB_ComplexNet
from load_data import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
args = parser.parse_args()
dataset = args.dataset

tinyimagenet_path = '../data'
figure_save_path = '../figure'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data and model
if 'mnist' in dataset:
    train_loader, test_loader, num_classes = load_mnist(dataset)
    model = MNIST_ComplexNet().to(device)
elif 'cifar' in dataset:
    train_loader, test_loader, num_classes = load_cifar(dataset)
    model = CIFAR_ComplexNet().to(device)
elif 'tinyimagenet' in dataset:
    train_loader, test_loader, num_classes = load_tinyimagenet(tinyimagenet_path)
    model = TinyImageNet_ComplexNet().to(device)
elif 'imdb' in dataset:
    train_loader, test_loader, num_classes = load_imdb()
    model = IMDB_ComplexNet().to(device)


spectral_norm_list = []
train_loss_list = []
test_loss_list = []

num_epochs = 100
lr = 0.01

for epoch in range(num_epochs):
    epoch = epoch + 1
    sn_prod = compute_spectral_norm_product(model)
    spectral_norm_list.append(sn_prod)
    train_loss_list.append(test(model, device, train_loader, epoch))
    test_loss_list.append(test(model, device, test_loader, epoch))

    print(spectral_norm_list[-1])
    print(train_loss_list[-1])
    print(test_loss_list[-1])

    train(model, device, train_loader, epoch, lr)

sn_prod = compute_spectral_norm_product(model)
spectral_norm_list.append(sn_prod)
train_loss_list.append(test(model, device, train_loader, epoch))
test_loss_list.append(test(model, device, test_loader, epoch))

excess_risk = []
for i in range(101):
    excess_risk.append(test_loss_list[i] - train_loss_list[i])
# plot
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(5,5))

x = range(101)
ax = fig.add_subplot(111)
lns1 = ax.plot(x[1:], excess_risk[1:], label='Excess risk', c='dodgerblue')
ax2 = ax.twinx()
lns2 = ax2.plot(x[1:], spectral_norm_list[1:], label='SN prod', c='red')

ax2.grid()

ax.set_ylabel('Excess risk',  fontsize=22)
ax.set_xlabel('Epoch',  fontsize=22)
ax2.set_ylabel('Spectral norm product',  fontsize=22)

plt.title(dataset, fontsize=30)

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, prop={'size': 15}, handlelength=1.5, framealpha=0, loc=0)

sns.despine(top=True, right=True, left=True, bottom=True)
plt.savefig(figure_save_path + '/excess_risk_snprod_' + dataset +'.pdf', dpi=600, format='pdf', bbox_inches='tight', transparent=True)
plt.show()