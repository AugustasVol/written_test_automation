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
        print(x.max())
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
        self.dropout = nn.Dropout(0.25)

        self.conv00 = nn.Conv2d(1, 15, (2,2), stride=(2,2))
        self.conv01 = nn.Conv2d(15, 16, (2,2), stride=(2,2))
        self.conv02 = nn.Conv2d(16, 16, (1,1), stride=(1,1))
        
        self.conv10 = nn.Conv2d(16, 32, (3,3), stride=(3,3))
        self.conv11 = nn.Conv2d(32,32, (2,2), stride=(1,1))
        self.conv12 = nn.Conv2d(32,32, (1,1), stride=(1,1))
        
        self.conv20 = nn.Conv2d(32, 16, (1,5), stride=(1,2))
        self.conv21 = nn.Conv2d(16, 16, (1,5), stride=(1,2))
        self.conv22 = nn.Conv2d(16, 6, (1,1), stride=(1,1))
        
        self.final_dense = nn.Linear(6,category_number)

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())

        self.train(False)
    def forward(self,x):
        x = self.dropout(x)
        
        x = F.relu(self.conv00(x))
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        
        x = F.relu(self.conv20(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))

        x = x.view(-1, 6)

        x = F.sigmoid(self.final_dense(x))
        
        return x
