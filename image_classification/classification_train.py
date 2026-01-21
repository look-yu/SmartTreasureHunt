''''# 导入PyTorch核心库
import torch
# 导入自定义模型模块（包含ConvEncoder和ConvDecoder）
import classification_model
# 导入训练引擎模块（包含train_step和val_step）
import classification_engine
# 导入torchvision的图像变换模块
import torchvision.transforms as T
# 导入自定义数据加载模块
import classification_data
# 导入配置文件参数
import classification_config
# 导入numpy用于数据处理
import numpy as np
# 导入进度条工具
from tqdm import tqdm
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim
# 导入自定义工具函数（如seed_everything）
from common import utils

# 主程序入口
if __name__ == "__main__":
    # 检测GPU可用性并设置设备
    if torch.cuda.is_available():
        device = "cuda"  # 优先使用GPU
    else:
        device = "cpu"  # 回退到CPU

    # 打印随机种子配置信息
    print("设置训练分类模型的随机数种子, seed = {}".format(classification_config.SEED))

    # 调用工具函数设置全局随机种子（确保可复现性）
    utils.seed_everything(classification_config.SEED)

    # 定义图像预处理流程
    transforms = T.Compose([
        T.Resize((64, 64)),  # 统一缩放到64x64分辨率
        T.ToTensor()  # 转换为PyTorch张量（范围[0,1]）
    ])

    # 数据集创建阶段
    print("------------ 正在创建数据集 ------------")
    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    full_dataset = classification_data.FolderDataset(
        classification_config.IMG_PATH,
        classification_config.FASHION_LABELS_PATH,
        transforms
    )

    # 计算训练集和验证集大小
    train_size = int(classification_config.TRAIN_RATIO * len(full_dataset))  # 75%训练
    val_size = len(full_dataset) - train_size  # 25%验证

    # 随机划分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 数据加载器配置阶段
    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")
    # 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=classification_config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    # 验证数据加载器（不打乱，完整加载）
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=classification_config.TEST_BATCH_SIZE
    )
    # 全量数据加载器（用于生成嵌入）
    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=classification_config.FULL_BATCH_SIZE
    )

    print("------------ 数据加载器创建完成 ------------")

    # 指定损失函数为交叉熵损失函数，适用于多分类任务
    loss_fn = nn.CrossEntropyLoss()

    # 初始化编码器和解码器
    classifier = classification_model.Classifier()  # 创建分类器实例

    # 将模型移动到指定设备（GPU/CPU）
    classifier.to(device)

    # 指定优化器为Adam优化器，优化对象是模型的所有参数，学习率为0.001
    optimizer = optim.AdamW(classifier.parameters(), lr=classification_config.LEARNING_RATE)

    # 初始化最佳损失值为极大值
    min_loss = 9999

    # 开始训练循环
    print("------------ 开始训练 ------------")

    # 使用tqdm进度条遍历预设的epoch数量
    for epoch in tqdm(range(classification_config.EPOCHS)):
        # 执行一个训练epoch
        train_loss = classification_engine.train_step(
            classifier, train_loader, loss_fn, optimizer, device=device
        )
        # 打印当前epoch的训练损失
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")

        # 执行验证步骤
        val_loss = classification_engine.val_step(
            classifier, val_loader, loss_fn, device=device
        )

        # 模型保存逻辑：当验证损失创新低时保存模型
        if val_loss < min_loss:
            print("验证集的损失减小了，保存新的最好的模型。")
            min_loss = val_loss
            # 保存编码器和解码器状态字典
            torch.save(classifier.state_dict(), classification_config.CLASSIFIER_MODEL_NAME)
        else:
            print("验证集的损失没有减小，不保存模型。")

        # 打印验证损失
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")

    # 训练结束提示
    print("\n==========> 训练结束 <==========\n")
'''
# 导入PyTorch核心库
import torch
# 导入自定义模型模块（包含ConvEncoder和ConvDecoder）
import classification_model
# 导入训练引擎模块（包含train_step和val_step）
import classification_engine
# 导入torchvision的图像变换模块
import torchvision.transforms as T
# 导入自定义数据加载模块
import classification_data
# 导入配置文件参数
import classification_config
# 导入numpy用于数据处理
import numpy as np
# 导入进度条工具
from tqdm import tqdm
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim

# 修复：直接实现seed_everything函数，避免导入冲突
def seed_everything(seed):
    """
    设置全局随机种子以确保实验可复现性
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况下
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

# 主程序入口
if __name__ == "__main__":
    # 检测GPU可用性并设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 打印随机种子配置信息
    print("设置训练分类模型的随机数种子, seed = {}".format(classification_config.SEED))

    # 调用种子设置函数（直接使用本地实现的版本）
    seed_everything(classification_config.SEED)

    # 定义图像预处理流程
    transforms = T.Compose([
        T.Resize((64, 64)),  # 统一缩放到64x64分辨率
        T.ToTensor()  # 转换为PyTorch张量（范围[0,1]）
    ])

    # 数据集创建阶段
    print("------------ 正在创建数据集 ------------")
    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    full_dataset = classification_data.FolderDataset(
        classification_config.IMG_PATH,
        classification_config.FASHION_LABELS_PATH,
        transforms
    )

    # 计算训练集和验证集大小
    train_size = int(classification_config.TRAIN_RATIO * len(full_dataset))  # 75%训练
    val_size = len(full_dataset) - train_size  # 25%验证

    # 随机划分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 数据加载器配置阶段
    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")
    # 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=classification_config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0  # Windows系统建议设置为0，避免多进程问题
    )
    # 验证数据加载器（不打乱，完整加载）
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=classification_config.TEST_BATCH_SIZE,
        num_workers=0
    )
    # 全量数据加载器（用于生成嵌入）
    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=classification_config.FULL_BATCH_SIZE,
        num_workers=0
    )

    print("------------ 数据加载器创建完成 ------------")
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")
    print(f"训练批次数量: {len(train_loader)}, 验证批次数量: {len(val_loader)}")

    # 指定损失函数为交叉熵损失函数，适用于多分类任务
    loss_fn = nn.CrossEntropyLoss()

    # 初始化编码器和解码器
    classifier = classification_model.Classifier()  # 创建分类器实例

    # 将模型移动到指定设备（GPU/CPU）
    classifier.to(device)
    print(f"模型已移至设备: {device}")

    # 指定优化器为Adam优化器，优化对象是模型的所有参数
    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=classification_config.LEARNING_RATE,
        weight_decay=1e-5  # 增加权重衰减防止过拟合
    )

    # 初始化最佳损失值为极大值
    min_loss = float('inf')  # 使用float('inf')更规范

    # 开始训练循环
    print("------------ 开始训练 ------------")

    # 使用tqdm进度条遍历预设的epoch数量
    for epoch in tqdm(range(classification_config.EPOCHS), desc="训练进度"):
        # 执行一个训练epoch
        train_loss = classification_engine.train_step(
            classifier, train_loader, loss_fn, optimizer, device=device
        )
        # 打印当前epoch的训练损失
        print(f"\nEpoch [{epoch + 1}/{classification_config.EPOCHS}]")
        print(f"训练损失: {train_loss:.4f}")

        # 执行验证步骤
        val_loss = classification_engine.val_step(
            classifier, val_loader, loss_fn, device=device
        )

        # 模型保存逻辑：当验证损失创新低时保存模型
        if val_loss < min_loss:
            print(f"验证损失下降: {min_loss:.4f} -> {val_loss:.4f}，保存最佳模型...")
            min_loss = val_loss
            # 保存编码器和解码器状态字典
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, classification_config.CLASSIFIER_MODEL_NAME)
        else:
            print(f"验证损失未下降: 当前{val_loss:.4f}，最佳{min_loss:.4f}")

        # 打印验证损失
        print(f"验证损失: {val_loss:.4f}")

    # 训练结束提示
    print("\n==========> 训练结束 <==========\n")
    print(f"最佳验证损失: {min_loss:.4f}")
    print(f"最佳模型已保存至: {classification_config.CLASSIFIER_MODEL_NAME}")