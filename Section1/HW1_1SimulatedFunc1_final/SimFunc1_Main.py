# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:53:38 2022

@author: reekm
"""

# import

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
torch.manual_seed(42)

from torch.utils.data import DataLoader, TensorDataset

# from model0 import Model0
# from model1 import Model1
# from model2 import Model2

from model0 import Model0
from model1 import Model1
from model2 import Model2

# initial Parameters
num_of_rows = 300
lr = 0.0004
gamma_lr_scheduler = 0.1 
weight_decay = 1e-4
criterion = nn.MSELoss()
optimizer = torch.optim.Adam
num_epochs =2500
criterion_name = "MSE_LOSS_"
optimizer_name = "ADAM_opt"
filename = criterion_name+ optimizer_name+".png"
grad_norm_name = "_grad_norm_name1_2.png"
result_folder_name = "result3/"

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



# defining Function (sin (5 pi x)) / 5 pi x

Y_func = lambda x : (torch.sin(5*torch.pi*x)) /(5*torch.pi*x) 
X= torch.unsqueeze(torch.linspace(-1,1,num_of_rows),dim=1)
Y = Y_func(X)

plt.figure(figsize=(10,4))
plt.plot(X, Y, color = "red")
plt.title('Non-Linear Function Plotting')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.savefig(result_folder_name+'func1_plot_1_1.png')


#  Creating DataSet

dataset = TensorDataset(X,Y)
data_loader = DataLoader(dataset,1,shuffle=True)

model_0 = Model0()
model_1 = Model1()
model_2 = Model2()


# Print Model Parameters
print("Model 0: ",get_n_params(model_0))
print("Model 1: ",get_n_params(model_1))
print("Model 2: ",get_n_params(model_2))




def evaluate(model,loss_fn, val_loader):
    outputs = [model.validation_step(batch,loss_fn) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_grad_norm(model):
    grad_all=0.0
    grad =0
    
    for p in model.parameters():
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy()**2).sum()
            
        grad_all+=grad
        
    grad_norm=grad_all ** 0.5
    return grad_norm


def fit(epochs, lr,wt_decay, model, data_loader, criterion,opt_func):
    history = []
    grad_norm_per_epoch={}
#     train_history = []
    optimizer = opt_func(model.parameters(), lr,weight_decay=wt_decay)
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in data_loader:
            loss = model.training_step(batch,criterion)
            loss.backward()
            optimizer.step()
            grad_norm_per_epoch[epoch] = get_grad_norm(model)
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model,criterion, data_loader)
        model.epoch_end(epoch, result)
        history.append(result)
#         res2 = evaluate2(model,train_loader)
#         train_history.append(res2)
    return history,grad_norm_per_epoch

# Permutation for different loss function and optimizer 
# class RMSLELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
        
#     def forward(self, pred, actual):
#         return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

# # criterion = [nn.MSELoss(),nn.CrossEntropyLoss(), RMSLELoss(),nn.L1Loss()]
# # criterion = [nn.MSELoss(),nn.CrossEntropyLoss()]
# criterion = [nn.CrossEntropyLoss()]
# # optimizer = [torch.optim.Adam,torch.optim.SGD]
# optimizer = [torch.optim.Adam]
# num_epochs =100
# # criterion_name = ["MSE_LOSS_","CROSS_ENTROPY_LOSS_","Root_MS_LOG_LOSS_","Mean_abs_loss_"]
# # criterion_name = ["MSE_LOSS_","CROSS_ENTROPY_LOSS_"]
# criterion_name = ["CROSS_ENTROPY_LOSS_"]
# optimizer_name = ["ADAM_opt"]
# lr = 0.0004

# # filename = criterion_name+ optimizer_name+".png"
# # filename


result_0 = evaluate(model_0,criterion,data_loader)
result_1 = evaluate(model_1,criterion,data_loader)
result_2 = evaluate(model_2,criterion,data_loader)

print(result_0,result_1,result_2)

print("MODEL 0")
history_0,g0 = fit(num_epochs, lr,weight_decay, model_0, data_loader,criterion,optimizer)
print("MODEL 1")
history_1,g1  = fit(num_epochs, lr,weight_decay, model_1, data_loader, criterion,optimizer)
print("MODEL 2")
history_2 ,g2 = fit(num_epochs, lr,weight_decay, model_2, data_loader, criterion,optimizer)

val_losses_0 = [r['val_loss'] for r in [result_0] + history_0]
val_losses_1 = [r['val_loss'] for r in [result_1] + history_1]
val_losses_2 = [r['val_loss'] for r in [result_2] + history_2]

val_losses_0_div_10 = [value for index,value in enumerate(val_losses_0) if (index % 10) ==0 ]
val_losses_1_div_10 = [value for index,value in enumerate(val_losses_1) if (index % 10) ==0 ]
val_losses_2_div_10 = [value for index,value in enumerate(val_losses_2) if (index % 10) ==0 ]


# Plotting grad norm

def plot_grad_norm(list_val,name):
    plt.figure(figsize=(10,10))
    plt.plot(g0.values())
    plt.xlabel('epoch')
    plt.ylabel('grad_norm')
    plt.title('grad_norm vs. epochs');
    
    plt.savefig(name)

# printing grad norm plots
# plot_grad_norm(g0.values(),"result_final/model0_"+ grad_norm_name)
# plot_grad_norm(g2.values(),"result_final/model1_"+ grad_norm_name)
# plot_grad_norm(g2.values(),"result_final/model2_"+ grad_norm_name)

plot_grad_norm(g0.values(),result_folder_name+"model0_"+ grad_norm_name)
plot_grad_norm(g1.values(),result_folder_name+"model1_"+ grad_norm_name)
plot_grad_norm(g2.values(),result_folder_name+"model2_"+ grad_norm_name)



# Plotting Loss

plt.figure(figsize=(10,10))
plt.plot(val_losses_0_div_10)
plt.plot(val_losses_1_div_10)
plt.plot(val_losses_2_div_10)
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.legend(['Model 0','Model 1','Model 2'])
plt.title('model_loss vs. epochs');

plt.savefig(result_folder_name+"loss_"+filename)

# Plotting original values vs predicted values

plt.figure(figsize=(10,10))
plt.plot(X,Y)
plt.plot(X,model_0(X).detach().numpy())
plt.plot(X,model_1(X).detach().numpy())
plt.plot(X,model_2(X).detach().numpy())

plt.xlabel('X- Independent Variable')
plt.ylabel('Original/prdicted Value')
plt.legend(['sin(5pix)/5pix','Model 0','Model 1','Model 2'])
plt.title('Actual and fitted Model');
plt.savefig(result_folder_name+"prediction_"+filename)
