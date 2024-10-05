# -*- coding: utf-8 -*-
import argparse
import csv
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import Dataset

from edgecons import AdmmSGD


# os.environ['NUMEXPR_MAX_THREADS'] = '24' ## specific for my system

formatter = '%(asctime)s [ECL] %(levelname)s :  %(message)s'
logging.basicConfig(level=logging.DEBUG, format=formatter)


def init_weights(m):
    if type(m) == nn.Linear:
        print("Weigths initialized")
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

class BaseDataset(data.Dataset):
    def __init__(self, data_path):
        super(BaseDataset, self).__init__()
        self.data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
        print('self.data', self.data.shape)
        self.n_classes = len(np.unique(self.data[:, -1]))
        self.input_shape = self.data.shape[1] - 1

    def name(self):
        return 'BaseDataset'

    def __getitem__(self, global_index):
        # select frame from sequence

        index = global_index
        inp = self.data[index][:-1].copy()
        cl = self.data[index][-1]
        return {
            'input': inp,
            'class': int(cl),
        }

    def __len__(self):
        return len(self.data)




import torch
import torch.nn as nn

class SCNN(nn.Module):
    def __init__(self, input_shape=41, n_classes=5, sign_size=32, cha_input=32, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2, load_weights=False, device="cuda"):
        super().__init__()
        # \\to DO:
        input_dim = input_shape
        output_dim = n_classes
        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden
 
        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)
 
        # 1st conv layer
        conv1 = nn.Conv1d(
            cha_input, 
            cha_input*K, 
            kernel_size=5, 
            stride=1, 
            padding=2,  
            groups=cha_input, 
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)
 
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)
 
        # 2nd conv layer
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)
 
        # 3rd conv layer
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)

 
        # 4th conv layer
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)
 
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
 
        self.flt = nn.Flatten()
 
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)
 
    def forward(self, x):
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))
        print(x.size())
 
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)
        print(x.size())
 
        x = nn.functional.relu(self.conv1(x))
        print(x.size())
 
        x = self.ave_po_c1(x)
        print(x.size())
 
        x = self.dropout_c2(x)
        print(x.size())
        x = nn.functional.relu(self.conv2(x))
        x_s = x
        print(x.size())
 
        x = self.dropout_c3(x)
        print(x.size())
        x = nn.functional.relu(self.conv3(x))
        print(x.size())
 
        x = self.conv4(x)
        print(x.size())
        x =  x + x_s
        print(x.size())
        x = nn.functional.relu(x)
        print(x.size())
 
        x = self.avg_po_c4(x)
        print(x.size())
 
        x = self.flt(x)
        print(x.size())
 
        x = self.dropout2(x)
        print(x.size())
        x = self.dense2(x)
        print(x.size())
        output = torch.log_softmax(x, dim=0)
        print(output.size())
        return output


class DatasetCSV(Dataset):
    def __init__(self, csv_file, transform=None, pca=None):
        self.df = pd.read_csv(csv_file)
        self.X = self.df.drop("class", axis=1)
        #self.X = self.X.drop("Unnamed: 0", axis=1)
        self.X = self.X.to_numpy().astype(np.float32)
        self.y = self.df["class"].to_numpy()  # .astype(np.int32)
        self.pca = None
        if pca is None:
            print('fitting')
            # self.pca = LatentDirichletAllocation(n_components=32)
            # self.pca.fit(self.X, self.y)
        else:
            self.pca = pca

        self.n_components = 32

        self.transform = transform
        self.cl_weights = np.zeros((5,), dtype=np.float32)
        val_c = self.df['class'].value_counts()
        for i in range(5):
            self.cl_weights[i] = float(len(self.df['class'])) / val_c[i]
        self.cl_weights = self.cl_weights / np.sum(self.cl_weights)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()


        localX = self.X[idx]
        # localX = self.pca.transform(localX[None])[0]
        localY = self.y[idx]

        # print(type(localX), type(localY))
        # print(type(localY))
        if self.transform:
           localX = torch.from_numpy(localX)
           localY = torch.tensor(localY) 
      

        return [localX, localY]

class NewData:

    def __init__(self, name, nodes, algorithm="admm", device="cpu", interval=6, offset=0, log_dir="/log/"):
        self.classes = ('0', '1', '2', '3', '4')
        self.train_mask_list = {
            "ALPHA": [True, True, True, True, True],
            # 
            #"BETA": [True, True, True, True, True],
            #"GAMMA": [True, True, True, True, True],
            #"DELTA": [True, True, True, True, True],
            #"OMEGA": [True, True, True, True, True]

        }
        self.logger = logging.getLogger(name)
        # self.model = Net2(60).to(device)
        self.train_set = DatasetCSV("data/train.csv", transform=True)

        self.model = SCNN(input_shape=41, n_classes=5, load_weights=False,
                            device=device).to(device)
        # self.model = Net4(0, 60, 10, [200, 100, 50]).to(device) #self.model = Net4(0, 60, 10, [200, 100, 50], p=0.1).to(device) #[200, 100, 50]

        # self.model = Net().to(device)
        print(self.model)
        self.device = device
        # algorithm = "admm"

        
        self.optimizer = AdmmSGD(name, nodes, device, self.model, interval, offset, 1000)
        self.test_set = DatasetCSV("data/test.csv", transform=True, pca=self.train_set.pca)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=500, shuffle=False, num_workers=2)
        self.mask_list = self.train_mask_list[name]
        self.name = name
        self.latest_epoch = 0
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        self.log_file_loss = open(log_dir + "/" + name + '_loss.csv', 'w')
        self.writer_loss = csv.writer(self.log_file_loss, lineterminator='\n')

        self.log_file_result = open(log_dir + "/" + name + '_result.csv', 'w')
        self.writer_result = csv.writer(self.log_file_result, lineterminator='\n')

    def train(self, max_epoch=50, batch_size=500, test_interval=10, num_workers=2):
        self.logger.info('Training start!!')
        decoder_criterion = nn.MSELoss()
        # criterion = nn.MSELoss()

        # Data Load
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.train_set.cl_weights).to(self.device))

        # Get how many nodes assigned to be trained for each class
        ClassNodeList = []
        for c in self.classes:
            NodesForClass = []
            for node in self.train_mask_list:
                # print(node[int(c)])
                if self.train_mask_list[node][int(c)]:
                    NodesForClass.append(node)
            ClassNodeList.append(NodesForClass)

        Mask = [False for i in self.train_set.y]

        # Create a Mask to select data that are for the node (if more nodes are assigned to the same class, it gets evenly divided between them)
        currentTrainMaskList = self.train_mask_list[self.name]
        for index, c in enumerate(ClassNodeList):
            if len(c) == 1 and currentTrainMaskList[index] == True:
                for maskIndex, i in enumerate(self.train_set.y):
                    if i == index:
                        Mask[maskIndex] = True
            if len(c) >= 2 and currentTrainMaskList[index] == True:
                lenOfEachChunk = int(sum(self.train_set.y == index) / len(c))
                StartofChunk = c.index(self.name) * lenOfEachChunk
                endOfChunk = (c.index(self.name) + 1) * lenOfEachChunk
                currentIndexOfChunk = 0
                # print(StartofChunk)
                # print(endOfChunk)
                for maskIndex, i in enumerate(self.train_set.y):
                    if i == index:
                        currentIndexOfChunk += 1
                        if currentIndexOfChunk >= StartofChunk and currentIndexOfChunk <= endOfChunk:
                            Mask[maskIndex] = True
        print("Samples for current node: ", sum(Mask))

        sampler = torch.utils.data.sampler.WeightedRandomSampler(Mask, len(self.train_set))
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, sampler=sampler,
                                                   num_workers=num_workers)

        for epoch in range(max_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            latest_diff = 0.0
            epc_cnt = 0
            start_time = time.time()

            self.model.train()
            for i, data in enumerate(train_loader, 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).long()


                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)

                running_loss += loss.item()
                # loss = loss.float()
                # zero the parameter gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update
                self.optimizer.update()
                epc_cnt += 1

            end_time = time.time()
            self.latest_epoch = epoch + 1
            latest_loss = running_loss / epc_cnt
            diff = self.optimizer.diff()
            latest_diff = diff.item()

            self.model.eval()
            with torch.no_grad():
                test_criterion = nn.CrossEntropyLoss(reduction="sum")
                # test_criterion = nn.MSELoss()
                test_loss = 0.0
                epc_cnt = 0
                for data in self.test_loader:
                    rows, labels = data
                    rows = rows.float()
                    # labels = labels.float()
                    rows = rows.to(self.device)
                    labels = labels.to(self.device).long()
                    outputs = self.model(rows)
                    loss = test_criterion(outputs, labels)
                    test_loss += loss.item()
                    epc_cnt += 1

            latest_test_loss = test_loss / epc_cnt / 100
            self.logger.info('[%03d] loss: %.3f/%.3f, diff: %.8f, time: %.2fsec' %
                             (self.latest_epoch, latest_loss, latest_test_loss, latest_diff, end_time - start_time))
            self.writer_loss.writerow([self.latest_epoch, latest_loss, latest_test_loss, latest_diff])

            if self.latest_epoch == 1 or self.latest_epoch % test_interval == 0:
                self.test()

        torch.save(self.model.state_dict(), "model.pth")
        self.logger.info('Finished Training')
        self.test()

        self.log_file_loss.close()
        self.log_file_result.close()
    def test(self):
        class_correct = list(0. for _ in range(5))
        class_total = list(0. for _ in range(5))

        with torch.no_grad():
            total = 0
            correct = 0
            targetsList = []
            predictionsList = []
            for data in self.test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images.float())
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                targetsList.extend(labels.data.tolist())
                predictionsList.extend(predicted.data.tolist())

                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Test:")
            print(metrics.classification_report(targetsList, predictionsList))
            print("Confusion Matrix:")
            print(metrics.confusion_matrix(targetsList, predictionsList))

        log_result = []
        log_result.append(self.latest_epoch)

        self.logger.info('Accuracy: %d %%' % (100 * correct / total))
        for i in range(5):
            class_result = 100 * class_correct[i] / class_total[i]
            self.logger.info('%5s : %2d %%' % (self.classes[i], class_result))
            log_result.append(class_result)

        self.writer_result.writerow(log_result)





def Newmain():
    parser = argparse.ArgumentParser(description='mnwmnist')
    parser.add_argument('-c', '--conf', required=True)
    parser.add_argument('-n', '--nodes', required=True)
    parser.add_argument('-a', '--algorithm', default="admm")
    args = parser.parse_args()

    with open(args.conf) as f:
        conf = json.load(f)
        name = conf["name"]
        interval = conf["interval"]
        offset = conf["offset"]
        device = conf["device"]

    with open(args.nodes) as f:
        conf = json.load(f)
        nodes = conf["nodes"]



    new_data = NewData(name, nodes, args.algorithm, device, interval, offset, "log/")
    new_data.train()


if __name__ == "__main__":
    Newmain()