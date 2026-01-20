#import DataLoader
from denoising_test import DataLoader
import torch
import transform

import denoising_engine
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#自定义相关组件
import denoising_model
import denoising_data
import denoising_config
from common import utils

def test(model, dataloader, device):
    import matplotlib.pyplot as plt
    dataiter = iter(train_loader)
    noisy_imgs, clean_imgs = next(dataiter)

    print("测试集 images 的形状:", noisy_imgs.shape)

    model = model.to(device)
    noisy_imgs = noisy_imgs.to(device)

    # 模型前向
    outputs = model(noisy_imgs)
    print('测试集输出结果的形状:', outputs.shape)

    # 转回 numpy 并移动通道到最后 (H,W,C)
    noisy_imgs_np = noisy_imgs.cpu().numpy()
    noisy_imgs_np = np.moveaxis(noisy_imgs_np, 1, -1)
    print('noisy_imgs np shape:', noisy_imgs_np.shape)

    outputs_np = outputs.detach().cpu().numpy()
    outputs_np = np.moveaxis(outputs_np, 1, -1)
    print('outputs np shape:', outputs_np.shape)

    # 原始干净图像
    original_images = clean_imgs.cpu().numpy()
    original_images = np.moveaxis(original_images, 1, -1)
    print('original_images shape:', original_images.shape)

    # 显示（最多 10 列）
    ncols = min(10, noisy_imgs_np.shape[0])
    fig, axes = plt.subplots(nrows=3, ncols=ncols, sharex=True, sharey=True, figsize=(12, 6))

    for imgs, ax_row in zip([noisy_imgs_np, outputs_np, original_images], axes):
        for img, ax in zip(imgs[:ncols], ax_row):
            ax.imshow(img)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

 
if __name__=='__main__':
 if torch.cuda.is_available():
    device='cuda'
 else:
    device='cpu'

    #打印随机种子配置信息
    print(torch.initial_seed())
    #调用工具函数设置全局随机种子
    torch.manual_seed(42)
    print(torch.initial_seed())

    #定义图像预处理流程
    transforms=T.Compose([T.Resize((68,68)),T.ToTensor])

    #打印日志，提示用户正在创建数据集
    print("--------------正在创建的数据集------------")

    #实例化完整数据集
    full_dataset= denoising_data.ImageDataset (image_dir ='../common/dataset/', transform=transform)
    train_size=int(0.75*len(full_dataset))
    val_size=len(full_dataset) -train_size
    #随机划分数据集
    train_dataset ,val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    #数据加载器配置阶段
    batch_size=32
    
    
    print('数据集创建完成，正在创建数据加载器...')
    #创建训练集数据加载器
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    #验证数据加载器
    val_loader=torch.utils.data.DataLoader(
        val_dataset,
        batch_size=denoising_config.BATCH_SIZE,
    )
    print('-----------数据加载器创建完成')

    #初始化自编码器用于去噪
    denoiser=denoising_model.ConvDenoiser().to(device)
    #指定损失函数
    loss_fn=nn.MSELoss()
    #将模型移到指定设备
    denoiser.to(device)
    #指定优化器
    optimizer=torch.optim.Adam(denoiser.parameters(), lr=denoising_config.LEARNING_RATE)
    #初始化最佳损失之为一个很大的数
    min_loss =9999
    #开始训练模型
    for epoch in range(denoising_config.NUM_EPOCHS):
        train_loss=denoising_utils.train_one_epoch(
            model=denoiser,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,

        )

    print(f'\n---------->epochs={epoch+1}, Train Loss: {train_loss:.4f}<---------')
    #执行验证步骤
    val_loss=denoising_utils.validate(
        model=denoiser,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
    )
    #模型保存逻辑，如果当前验证损失小于最小损失，则保存模型
    if val_loss < min_loss:
        print(f'验证损失从 {min_loss:.4f} 降低到 {val_loss:.4f}，保存模型...')
        min_loss=val_loss
        torch.save(denoiser.state_dict(),'denoising_model.pth')

    else:
        print(f'验证损失未改善，当前最小损失仍为 {min_loss:.4f}')
        #打印损失
        print(f'Epochs [{epoch+1}/{denoising_config.NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')

        print('训练完成！')

        print('本次训练的去噪模型训练结果如下')

        test(denoiser,val_loader,device)

        print('---------->从磁盘加载模型<------------')

        load_denoiser =denoising_model.ConvDenoiser().to(device)
        load_denoiser.load_state_dict(torch.load('denoising_model.pth'))

        print('从磁盘加载的去噪模型测试结果如下')
        test(load_denoiser,val_loader,device)