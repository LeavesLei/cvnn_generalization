import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from numpy import linalg as LA

num_epochs =100

def train(model, device, train_loader, epoch, lr=0.1):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer = optim.SGD(model.parameters(), lr=learning_rate(lr, epoch), momentum=0.9)


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target =data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    len(train_loader), loss.item(), 100.*correct/total))
        sys.stdout.flush()
            
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        batch_num = batch_num + 1
        data, target = data.to(device).type(torch.complex64), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        test_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        #print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    return test_loss / batch_num


# For Tiny ImageNet and IMDB
def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 100):
        optim_factor = 3
    elif(epoch > 80):
        optim_factor = 2
    elif(epoch > 40):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)

# For other datasets
#def learning_rate(init, epoch):
#    return init


def spectral_norm_for_complex_matrix(real_mat, imaginary_mat):
    hermitian_mat = np.matmul(real_mat.T, real_mat) + np.matmul(imaginary_mat.T, imaginary_mat) + 1j*(np.matmul(real_mat.T, imaginary_mat) - np.matmul(imaginary_mat.T, real_mat))
    return np.sqrt(np.max(LA.eigvalsh(hermitian_mat)))


def compute_spectral_norm_product(model):
    real_mat_list = []
    imag_mat_list = []
    norm_list = []

    for param in model.named_parameters():
        weight = param[1].detach().cpu().numpy()
        if "conv_r.weight" in param[0]:
            real_mat_list.append(np.reshape(weight, (weight.shape[0], -1)))
        elif "conv_i.weight" in param[0]:
            imag_mat_list.append(np.reshape(weight, (weight.shape[0], -1)))
        elif "fc_r.weight"  in param[0]:
            real_mat_list.append(weight)
        elif "fc_i.weight"  in param[0]:
            imag_mat_list.append(weight)
        else:
            continue

    for i in range(len(real_mat_list)):
        sn = spectral_norm_for_complex_matrix(real_mat_list[i], imag_mat_list[i])
        norm_list.append(sn)
    return np.prod(norm_list)