import os
import torch
from config import cfg
from datasets.make_dataloader import make_dataloader
from tools.train import train
from tools.test import test, test_prune

# 设置CUDA可见的设备
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

if __name__ == '__main__':
    # 创建训练和验证集的数据加载器以及一些其他参数
    train_loader, val_loader, num_classes = make_dataloader(cfg)

    # 如果模型模式是训练
    if cfg.MODEL.MODE == 'train':
        # 训练模型
        train(train_loader, num_classes, deploy_flag=False)

        # 在训练后进行测试
        with torch.no_grad():
            # 测试模型（部署标志设置为True）
            test(val_loader, num_classes, deploy_flag=True)

            # 进行修剪前的测试
            test_prune(val_loader, num_classes, deploy_flag=False)

            # 进行修剪后的测试
            test_prune(val_loader, num_classes, deploy_flag=True, test_rep_prune=True)

    # 如果模型模式是评估
    if cfg.MODEL.MODE == 'evaluate':
        # 在评估模式下进行测试
        with torch.no_grad():
            # 测试模型（部署标志设置为True）
            test(val_loader, num_classes, deploy_flag=True)

            # 进行修剪前的测试
            test_prune(val_loader, num_classes, deploy_flag=False)

            # 进行修剪后的测试
            test_prune(val_loader, num_classes, deploy_flag=True, test_rep_prune=True)
