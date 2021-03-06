import os
import time
import torch 
import numpy as np
from tqdm import tqdm 
from logger import Logger
import torch.optim as optim
from datetime import datetime 
from lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from torchvision.models import resnet18
from dataset_loader import  gpu_transformer
from model import SimSiam
from knn_monitor import knn_monitor as accuracy_monitor
from dataset_loader import get_train_mem_test_dataloaders

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")



uid = 'simsiam'
dataset_name = 'cifar10'
data_dir = 'dataset'
ckpt_dir = "./ckpt/"+str(datetime.now().strftime('%m%d%H%M%S'))
log_dir = "runs/"+str(datetime.now().strftime('%m%d%H%M%S'))

#create dataset folder 
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Setup asset directories
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
logger = Logger(log_dir=log_dir, tensorboard=True, matplotlib=True)



#hyperparams
warmup_epochs = 50 
warmup_lr = 0
base_lr = 0.03
final_lr = 0.00001
num_epochs = 800 # this parameter influence the lr decay
stop_at_epoch = 50 # has to be smaller than num_epochs
batch_size = 128
knn_interval =  3
knn_k = 80

image_size = (32,32)

train_loader, memory_loader, test_loader = get_train_mem_test_dataloaders(
                dataset="cifar10", 
                data_dir="./dataset",
                batch_size=batch_size,
                num_workers=4, 
                download=True )

train_transform , test_transform = gpu_transformer(image_size)


# model
model = SimSiam().to(device)

# optimizer
momentum = 0.9
weight_decay = 0.0005


predictor_prefix = ('module.predictor', 'predictor')
parameters = [{
    'name': 'base',
    'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
    'lr': base_lr
},{
    'name': 'predictor',
    'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
    'lr': base_lr
    }]




optimizer = torch.optim.SGD(parameters, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

scheduler = LR_Scheduler(
    optimizer, warmup_epochs, warmup_lr*batch_size/256,

    num_epochs, base_lr*batch_size/256, final_lr*batch_size/256, 
    len(train_loader),
    constant_predictor_lr=True 
    )

min_loss = np.inf
global_progress = tqdm(range(0, stop_at_epoch), desc=f'Training')
data_dict = {"loss": 100}


for epoch in global_progress:
    model.train()   
    local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
    
    for idx, (image, label) in enumerate(local_progress):
        image = image.to(device)
        aug_image = train_transform(image).to(device)
        model.zero_grad()
        ret = model.forward(image,aug_image)
        loss = ret 
        data_dict['loss'] = loss.item() 
        loss.backward()
        optimizer.step()
        scheduler.step()
        data_dict.update({'lr': scheduler.get_last_lr()})
        local_progress.set_postfix(data_dict)
        logger.update_scalers(data_dict)
    
    current_loss = data_dict['loss']

    '''if epoch % knn_interval == 0: 
        accuracy = accuracy_monitor(model.backbone, memory_loader, test_loader, 'cpu', hide_progress=True) 
        data_dict['accuracy'] = accuracy
    '''

    global_progress.set_postfix(data_dict)
    logger.update_scalers(data_dict)
    
    model_path = os.path.join(ckpt_dir, f"{uid}_{datetime.now().strftime('%m%d%H%M%S')}.pth")

    if min_loss > current_loss:
        min_loss = current_loss
        
        torch.save({
        'epoch':epoch+1,
        'state_dict': model.state_dict() }, model_path)
