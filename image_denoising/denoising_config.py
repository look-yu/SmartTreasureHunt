IMG_PATH = "data/images/noisy/"
IMG_HEIGHT = 512
IMG_WIDTH = 512

#随机性和数据划分配置
#训练集划分比例
TRAIN_SPLIT = 0.75
#随机种子，确保结果可复现
RANDOM_SEED = 42
#数据增强配置
NOISE_FACTOR= 0.5
SHUFFLe_BUFFER_SIZE = 1000

LEARNING_RATE = 1e-3
EPOCHS = 30
TRAINING_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

#-------------模型配置-------------
PACKAGE_NAME = "image_denoising"
# 编码器权重保存路径（需写权限）
DENOISER_MODEL_NAME = "denoiser.pt"
