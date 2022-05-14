from my_dataset import MahjongGBDataset,DataLoaderX
# from torch.utils.data import DataLoader
import model_test
import torch.nn.functional as F
import torch
import time
import os

if __name__ == '__main__':
    logdir = 'code/log/'
    resnet_depth=34
    timestamp=int(time.time())
    os.mkdir(logdir + f'checkpoint_{resnet_depth}_{timestamp}')
    
    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoaderX(dataset = trainDataset, batch_size = batchSize, shuffle = False)
    vloader = DataLoaderX(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    
    # Load model
    model = model_test.resnet34(True, 0.5, (147,235)).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    
    # Train and validate
    for e in range(10):
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
        torch.save(model.state_dict(), logdir + f'checkpoint_{resnet_depth}_{timestamp}/{e}.pkl')
        print(logdir + f'checkpoint_{resnet_depth}_{timestamp}/{e}.pkl saved')