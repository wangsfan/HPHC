"""
测试 mmsegmentation Swin-Large 模型在完整 VOC 数据集（21类）上的分割效果
基于 train_mmseg_swin_large_voc_saved.py，加载指定权重文件进行测试评估
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import logging
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import math

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加 mmsegmentation 路径
mmseg_dir = os.path.join(current_dir, 'Swin-Transformer-Semantic-Segmentation')
sys.path.insert(0, mmseg_dir)

# 添加 WILSON 路径
wilson_dir = os.path.join(current_dir, 'WILSON')
sys.path.insert(0, wilson_dir)

# 导入 mmsegmentation 模块
from mmcv import Config
from mmseg.models import build_segmentor
from mmseg.apis import set_random_seed

# 导入 WILSON 数据加载器
from data_loader import VOCSegmentation

# 导入评估指标
from metrics import StreamSegMetrics


# ==================== 数据转换 ====================
class VOCTransform:
    """VOC 数据集转换，适配 mmsegmentation 格式（与 train_mmseg_swin_large_voc_saved.py 一致）"""
    def __init__(self, crop_size=512, train=True):
        self.crop_size = crop_size
        self.train = train
        
        # 图像归一化（ImageNet 标准，用于 [0, 1] 范围的 tensor）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __call__(self, img, target):
        """
        Args:
            img: PIL Image
            target: PIL Image (分割标签)
        
        Returns:
            img: torch.Tensor [3, H, W] (归一化后的图像)
            target: torch.Tensor [H, W] (标签，值为 0-20)
        """
        # 确保 target 是 PIL Image
        if not isinstance(target, Image.Image):
            target = Image.fromarray(np.array(target))
        
        # 确保标签值在有效范围内 (0-20)
        target_array = np.array(target)
        target_array = np.clip(target_array, 0, 20)
        target = Image.fromarray(target_array.astype(np.uint8))
        
        if self.train:
            # 训练时：随机缩放裁剪 + 随机水平翻转（测试时不需要）
            # 这里为了兼容性保留，但测试时不会使用
            pass
        else:
            # 验证/测试时：调整大小 + 中心裁剪
            h, w = img.size
            if h < w:
                new_h, new_w = self.crop_size, int(w * self.crop_size / h)
            else:
                new_h, new_w = int(h * self.crop_size / w), self.crop_size
            
            img = TF.resize(img, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
            target = TF.resize(target, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
            
            # 中心裁剪
            img = TF.center_crop(img, (self.crop_size, self.crop_size))
            target = TF.center_crop(target, (self.crop_size, self.crop_size))
        
        # 转换为 tensor（图像转换为 [0, 1] 范围）
        img = TF.to_tensor(img)  # [3, H, W], 值范围 [0, 1]
        target = torch.from_numpy(np.array(target)).long()  # [H, W]
        
        # 归一化图像（使用 ImageNet 标准归一化）
        img = self.normalize(img)
        
        return img, target


# ==================== 测试配置 ====================
class TestConfig:
    """测试配置类"""
    def __init__(self):
        # 数据集配置
        self.data_root = os.path.join(current_dir, 'WILSON', 'data')
        self.crop_size = 512  # mmsegmentation 标准尺寸
        self.crop_size_val = 512
        self.batch_size = 1
        self.num_workers = 4
        self.n_classes = 21  # VOC 完整数据集有 21 类
        
        # 模型配置
        self.model_size = 'large'  # 'tiny', 'small', 'base', 'large'
        
        # 检查点路径
        self.checkpoint_path = os.path.join(
            current_dir,
            'outputs/mmseg_swin_large_voc_21class/checkpoint_iter_17000_best_0.6945_best.pth'
        )
        
        # 输出目录
        self.output_dir = os.path.join(
            current_dir, 
            'outputs', 
            f'mmseg_swin_{self.model_size}_voc_21class_test'
        )
        
        # 随机种子
        self.seed = 1234
        
        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_cuda = torch.cuda.is_available()


def test(model, test_loader, metrics, device):
    """测试函数"""
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', leave=False)
        for images, labels, l1h in pbar:
            images = images.to(device)
            labels = labels.to(device).long()
            
            # 准备数据格式
            img_metas = [{
                'img_shape': images.shape[2:],
                'ori_shape': images.shape[2:],
                'pad_shape': images.shape[2:],
                'scale_factor': 1.0,
                'flip': False
            } for _ in range(images.size(0))]
            
            # 前向传播（测试模式）
            seg_logits = model.encode_decode(images, img_metas)
            
            # 获取预测结果（argmax）
            # seg_logits 是 [B, C, H, W]，需要取 argmax 得到 [B, H, W]
            _, predictions = seg_logits.max(dim=1)  # [B, H, W]
            
            # 转换为numpy并更新评估指标
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            metrics.update(labels_np, predictions_np)
    
    # 计算评估得分
    score = metrics.get_results()
    
    return score


def load_checkpoint(model, checkpoint_path, device, logger=None):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        error_msg = f"✗ 检查点文件不存在: {checkpoint_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if logger:
        logger.info(f"加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_val_iou = checkpoint.get('best_val_iou', None)
    else:
        model.load_state_dict(checkpoint)
        epoch = 0
        best_val_iou = None
    
    if logger:
        logger.info(f"✓ 检查点加载成功")
        logger.info(f"  检查点epoch: {epoch}")
        if best_val_iou is not None:
            logger.info(f"  检查点IoU: {best_val_iou:.4f}")
    
    return epoch, best_val_iou


# ==================== 主测试函数 ====================
def main(config=None):
    """主测试函数"""
    print("=" * 70)
    print("mmsegmentation Swin-Large 测试 - VOC 完整数据集（21类）")
    print("=" * 70)
    
    if config is None:
        config = TestConfig()
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(config.output_dir, 'test.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("测试配置:")
    logger.info(f"  模型: Swin-Large (mmsegmentation)")
    logger.info(f"  数据集: VOC 完整数据集（21类）")
    logger.info(f"  检查点: {config.checkpoint_path}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  设备: {config.device}")
    logger.info(f"  输出目录: {config.output_dir}")
    logger.info("=" * 70)
    
    # ==================== 加载数据集 ====================
    logger.info("\n加载数据集...")
    
    try:
        # 创建数据转换
        test_transform = VOCTransform(crop_size=config.crop_size_val, train=False)
        
        # 加载完整 VOC 数据集（21类）
        test_dataset = VOCSegmentation(
            root=config.data_root,
            train=False,  # 使用验证集作为测试集
            transform=test_transform,
            indices=None,  # 使用所有样本
            as_coco=False,
            saliency=False,
            pseudo=None
        )
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        logger.info(f"✓ 数据集加载成功")
        logger.info(f"  测试集: {len(test_dataset)} 样本")
        logger.info(f"  类别数: {config.n_classes}")
        logger.info(f"  图像尺寸: {config.crop_size_val}x{config.crop_size_val}")
    except Exception as e:
        logger.error(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 加载模型 ====================
    logger.info("\n加载模型...")
    
    # 加载 mmsegmentation 配置文件
    config_file = os.path.join(
        mmseg_dir, 
        'configs/swin/upernet_swin_large_patch4_window7_512x512_20k_voc12aug.py'
    )
    
    if not os.path.exists(config_file):
        logger.error(f"✗ 配置文件不存在: {config_file}")
        return
    
    logger.info(f"✓ 配置文件: {config_file}")
    
    # 加载配置
    cfg = Config.fromfile(config_file)
    
    # 更新类别数（21类）
    cfg.model.decode_head.num_classes = config.n_classes
    cfg.model.auxiliary_head.num_classes = config.n_classes
    
    # 不加载预训练模型（我们将从检查点加载）
    cfg.model.pretrained = None
    
    # 构建模型
    logger.info("构建模型...")
    try:
        model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        logger.info("✓ 模型构建成功")
        
        # 打印模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  可训练参数数: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 移动到指定设备
    if config.use_cuda:
        logger.info("移动模型到 GPU...")
        model = model.cuda()
        logger.info("✓ 模型已移动到 GPU")
    else:
        logger.info("使用 CPU")
        model = model.cpu()
    
    # ==================== 加载检查点 ====================
    logger.info("\n加载检查点...")
    try:
        epoch, best_val_iou = load_checkpoint(
            model, config.checkpoint_path, config.device, logger
        )
    except Exception as e:
        logger.error(f"✗ 加载检查点失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 创建评估指标对象 ====================
    logger.info("\n创建评估指标对象...")
    test_metrics = StreamSegMetrics(n_classes=config.n_classes)
    logger.info(f"✓ 评估指标对象创建成功 (类别数: {config.n_classes})")
    
    # ==================== 测试评估 ====================
    logger.info("\n" + "=" * 70)
    logger.info("在测试集上进行评估...")
    logger.info("=" * 70)
    
    test_score = test(model, test_loader, test_metrics, config.device)
    
    logger.info("\n" + "=" * 70)
    logger.info("测试集评估结果:")
    logger.info("=" * 70)
    logger.info(f"  Overall Accuracy: {test_score['Overall Acc']:.4f}")
    logger.info(f"  Mean Accuracy: {test_score['Mean Acc']:.4f}")
    logger.info(f"  Mean Precision: {test_score['Mean Prec']:.4f}")
    logger.info(f"  Mean IoU: {test_score['Mean IoU']:.4f}")
    
    logger.info("\n测试集各类别IoU:")
    for class_id, iou in test_score['Class IoU'].items():
        if iou != "X":
            logger.info(f"  类别 {class_id}: {iou:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)
    logger.info(f"测试结果保存在: {log_file}")
    
    # 同时打印到控制台
    print("\n" + "=" * 70)
    print("测试集评估结果:")
    print("=" * 70)
    print(f"  Overall Accuracy: {test_score['Overall Acc']:.4f}")
    print(f"  Mean Accuracy: {test_score['Mean Acc']:.4f}")
    print(f"  Mean Precision: {test_score['Mean Prec']:.4f}")
    print(f"  Mean IoU: {test_score['Mean IoU']:.4f}")
    print("\n测试集各类别IoU:")
    for class_id, iou in test_score['Class IoU'].items():
        if iou != "X":
            print(f"  类别 {class_id}: {iou:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 mmsegmentation Swin-Large 模型 (VOC 完整数据集 21类)')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/shl/pps/WILSS/outputs/mmseg_swin_large_voc_21class/checkpoint_iter_17000_best_0.6945_best.pth',
                        help='检查点路径')
    
    args = parser.parse_args()
    
    # 创建配置并设置 checkpoint
    config = TestConfig()
    if args.checkpoint is not None:
        config.checkpoint_path = args.checkpoint
    
    main(config)


