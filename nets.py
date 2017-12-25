import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class net_base:
    def trainer(self, x,y, epochs = 1):
        
        self.train(True)
        
        for i in range(epochs):
            print(i)
            self.optimizer.zero_grad()   # zero the gradient buffers
        
            output = self(x)
            loss = self.loss_function(output, y)
            loss.backward()
            print(loss)
        
            self.optimizer.step()    # Does the update
        
        self.train(False)
    def numpy_forward(self, x):

        if x.dtype == np.uint8:
            x = x / 255

        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = autograd.Variable(x)
        output = self(x)
        
        return output.data.numpy()
    def numpy_train(self,x,y, epochs = 1):

        if x.dtype == np.uint8:
            x = x / 255

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        x = autograd.Variable(x)
        y = autograd.Variable(y)
        
        self.trainer(x,y, epochs = epochs)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
    def save_weights(self,path):
        torch.save(self.state_dict(), path)



class answer_model(nn.Module, net_base):
    def __init__(self, category_number = 6):
        super(answer_model, self).__init__()

        self.conv1 = nn.Conv2d(1,  8, (3,3), stride=(1,1) )
        self.conv2 = nn.Conv2d(8,  8, (5,5), stride=(2,2) )
        self.conv3 = nn.Conv2d(8, 16, (5,5), stride=(2,2) )
        self.conv4 = nn.Conv2d(16,16, (5,5), stride=(1,2) )
        self.conv5 = nn.Conv2d(16,32, (1,3), stride=(1,2) )

        self.dropout1 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256 ,32)
        
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(32,32)

        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(32,32)

        self.final_dense = nn.Linear(32,category_number)

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())

        self.train(False)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.relu(self.conv3(x))
        #print(x.size())
        x = F.relu(self.conv4(x))
        #print(x.size())
        x = F.relu(self.conv5(x))

        #print(x.size())

        x = x.view(-1, 256)

        x = F.relu(self.dropout1(self.dense1(x)))
        x = F.relu(self.dropout2(self.dense2(x)))
        x = F.relu(self.dropout3(self.dense3(x)))

        x = F.sigmoid(self.final_dense(x))
        return x

