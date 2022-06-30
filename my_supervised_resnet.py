from my_dataset import MahjongGBDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import model_test
import torch.nn.functional as F
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
 
log_writer = SummaryWriter()
n_gpu=8

if __name__ == '__main__':
    # 集群上跑用这个 logdir = '/code/log/'
    logdir = '/code/log/' 
    resnet_depth=34
    timestamp=int(time.time())
    os.mkdir(logdir + f'checkpoint_{resnet_depth}_{timestamp}')
    
    # Load dataset
    splitRatio = 0.7
    batchSize = 1024*n_gpu
    trainDataset = MahjongGBDataset(0.2, 0.9, True)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    validateDataset = MahjongGBDataset(0.9, 1, False)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # debug
    # for i in range(len(validateDataset)):
    #     a=validateDataset[i]
    #     print(i)
        

    # Load model
    data_dir='/code/log/checkpoint_34_1656597631/14.pkl'
    model = model_test.resnet34(True, 0, (38,235)).to('cuda')
    model.load_state_dict(torch.load(data_dir))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # Train and validate
    for e in range(50):
        print('Epoch', e)
        
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(model.module.state_dict(), logdir + f'checkpoint_{resnet_depth}_{timestamp}/{e}.pkl')
        print(logdir + f'checkpoint_{resnet_depth}_{timestamp}/{e}.pkl saved')
        print('Run validation:')
        
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
        scheduler.step(acc)
        log_writer.add_scalar('Accuracy/valid', float(acc), e)
        