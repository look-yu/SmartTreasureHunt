__all__ = ["train_step", "val_step",'create_embedding']

import device
#导入PyTorch核心库和神经网络模块
import torch

def train_step(model, train_loader, optimizer, scheduler, num_epochs):
    #执行一个完整的训练迭代
    #设置为训练模式
    total_loss = 0
    num_batches=0

    #便利数据加载器中的所有批次
    for train_img,targetr in train_loader:
        train_img=train_img.to(torch.float32)
        targetr=targetr.to(torch.float32)
        train_img = train_img.cuda()
        targetr = targetr.to(device)

        #前向传播
        outputs = model(train_img)
        loss = torch.nn.functional.mse_loss(outputs, targetr)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches+=1

    return total_loss/num_batches
def val_step(model, val_loader, optimizer, scheduler, num_epochs):
    #执行一个完整的验证迭代
    #model.eval()  #设置为评估模式
    total_loss = 0
    num_batches=0

    with torch.no_grad():  #禁用梯度计算以节省内存
        for val_img,targetr in val_loader:
            val_img=val_img.to(torch.float32)
            targetr=targetr.to(torch.float32)
            val_img = val_img.cuda()
            targetr = targetr.to(device)

            #前向传播
            outputs = model(val_img)
            loss = torch.nn.functional.mse_loss(outputs, targetr)

            total_loss += loss.item()
            num_batches+=1

    return total_loss/num_batches