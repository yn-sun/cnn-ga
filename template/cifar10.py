"""
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_loader
import os
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #generated_init


    def forward(self, x):
        #generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self):
        data_dir = os.path.expanduser('./dataset')
        trainloader, validate_loader = data_loader.get_train_valid_loader(data_dir, batch_size=128, augment=True, valid_size=0.1, shuffle=True, random_seed=2312390, show_sample=False, num_workers=1, pin_memory=True)
        #testloader = data_loader.get_test_loader(data_dir, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        net = EvoCNNModel()
        cudnn.benchmark = True
        net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.trainloader = trainloader
        self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]
        #self.testloader = testloader
        #self.log_record(net, first_time=True)
        #self.log_record('+'*50, first_time=False)

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch):
        self.net.train()
        if epoch ==0: lr = 0.01
        if epoch > 0: lr = 0.1;
        if epoch > 148: lr = 0.01
        if epoch > 248: lr = 0.001
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-4)
        running_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        self.log_record('Train-Epoch:%3d,  Loss: %.3f, Acc:%.3f'% (epoch+1, running_loss/total, (correct/total)))

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct/total > self.best_acc:
            self.best_acc = correct/total
            #print('*'*100, self.best_acc)
        self.log_record('Validate-Loss:%.3f, Acc:%.3f'%(test_loss/total, correct/total))


    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        for p in range(total_epoch):
            self.train(p)
            self.test(total_epoch)
        return self.best_acc


class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        m = TrainModel()
        try:
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            best_acc = m.process()
            #import random
            #best_acc = random.random()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.3f'%best_acc)

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s=%.5f\n'%(file_id, best_acc))
            f.flush()
            f.close()
"""


