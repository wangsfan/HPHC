"""
使用 mmsegmentation 的 Swin-Tiny 模型训练 VOC 15-5 Step1 分阶段分割任务
基于 Step0 最佳模型（16类），扩展分类头到 21 类并训练新增的5类

参考 Step0 的配置：
- 模型：Swin-Tiny + UPerNet
- 优化器：AdamW (lr=0.00006)
- 学习率调度：Poly (warmup_iters=1500)
- Batch Size: 8
- 训练迭代：40k iterations（或50 epochs）
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import logging

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
from mmcv.runner import build_optimizer
from mmseg.models import build_segmentor
from mmseg.apis import set_random_seed

# 导入 WILSON 数据加载器
from data_loader import create_dataloaders, get_task_dict, get_task_labels
from load_voc_dataset import VOCDatasetConfig

# 导入评估指标
from metrics import StreamSegMetrics


# ==================== 训练配置 ====================
class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        # 基本训练参数（与 Step0 保持一致）
        self.max_epochs = 50  # 可以设置为30或更多，支持从检查点恢复继续训练
        self.base_lr = 0.00006  # Swin-Tiny 推荐学习率（与 Step0 一致）
        self.batch_size = 8  # Swin-Tiny 使用 batch_size=8（与 Step0 一致）
        self.num_workers = 4
        self.eval_interval = 1  # 每几个epoch验证一次（epoch级别的验证）
        self.eval_interval_iters = 500  # 每多少次迭代验证一次（迭代级别的验证，与mmsegmentation一致）
        
        # 数据集配置
        self.task = '15-5'
        self.step = 1  # Step1
        self.data_root = os.path.join(current_dir, 'WILSON', 'data')
        self.crop_size = 512  # mmsegmentation 标准尺寸
        self.crop_size_val = 512
        
        # 模型配置（与 Step0 保持一致，使用 Swin-Tiny）
        self.model_size = 'tiny'  # 'tiny', 'small', 'base', 'large' - 与 Step0 一致
        self.pretrained_model = os.path.join(
            mmseg_dir, 
            'checkpoints/swin_tiny_patch4_window7_224.pth'
        )
        
        # Step0 最佳模型路径（16类训练的最佳模型）
        self.step0_checkpoint = os.path.join(
            current_dir,
            'outputs/mmseg_swin_tiny_voc_15-5_step0/checkpoint_iter_36500_best_best.pth'
        )
        
        # 优化器配置（AdamW，与 mmsegmentation 配置一致）
        self.optimizer_type = 'AdamW'
        self.weight_decay = 0.01
        self.betas = (0.9, 0.999)
        
        # 学习率调度器配置（多项式衰减，与 mmsegmentation 配置一致）
        self.lr_scheduler = 'poly'  # 'poly', 'cosine', 'step'
        self.power = 1.0
        self.min_lr = 0.0
        self.warmup_iters = 1500
        self.warmup_ratio = 1e-6
        
        # 输出目录
        self.output_dir = os.path.join(
            current_dir, 
            'outputs', 
            f'mmseg_swin_{self.model_size}_voc_{self.task}_step{self.step}'
        )
        
        # 恢复训练
        self.resume = None  # 检查点路径
        
        # 随机种子
        self.seed = 1234
        
        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_cuda = torch.cuda.is_available()
        
        # 保存检查点
        self.save_best = True
        self.save_last = True


# ==================== 学习率调度器 ====================
def update_learning_rate_poly(optimizer, current_iter, max_iters, base_lr, 
                              power=1.0, min_lr=0.0, warmup_iters=0, warmup_ratio=1e-6):
    """多项式学习率调度（与 mmsegmentation 一致）"""
    if current_iter < warmup_iters:
        # Warmup 阶段
        lr = base_lr * (warmup_ratio + (1 - warmup_ratio) * current_iter / warmup_iters)
    else:
        # 多项式衰减
        lr = base_lr * ((1 - current_iter / max_iters) ** power)
        lr = max(lr, min_lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


# ==================== 扩充分类头并复制权重 ====================
def expand_classification_head(model, step0_state_dict, old_num_classes=16, new_num_classes=21, logger=None):
    """
    扩展分类头从 old_num_classes 到 new_num_classes，并复制权重
    
    Args:
        model: 新模型（21类）
        step0_state_dict: Step0 模型的状态字典
        old_num_classes: 旧类别数（16）
        new_num_classes: 新类别数（21）
        logger: 日志记录器
    """
    if logger:
        logger.info(f"\n扩展分类头: {old_num_classes}类 -> {new_num_classes}类")
    
    # 扩展 decode_head 分类头
    if hasattr(model, 'decode_head') and model.decode_head is not None:
        decode_head = model.decode_head
        
        # 查找分类头的 conv_seg 层
        if hasattr(decode_head, 'conv_seg'):
            conv_seg = decode_head.conv_seg
            
            # 获取旧权重
            old_weight_key = 'decode_head.conv_seg.weight'
            if old_weight_key in step0_state_dict:
                old_weight = step0_state_dict[old_weight_key]  # [old_num_classes, channels, 1, 1]
                
                # 创建新权重
                channels = old_weight.shape[1]
                new_weight = torch.zeros(new_num_classes, channels, 1, 1, device=old_weight.device, dtype=old_weight.dtype)
                
                # 复制旧类别权重 (0-15)
                new_weight[:old_num_classes] = old_weight
                
                # 对于新类别 (16-20)，复制 step0 最佳模型的权重
                # 根据用户要求：除了背景类复制背景类权重外，其他都用最佳模型文件的权重
                # 新类别使用 step0 最佳模型中所有类别的平均权重
                # 使用所有旧类别的平均权重来初始化新类别
                avg_weight = old_weight.mean(dim=0, keepdim=True)  # [1, channels, 1, 1]
                
                # 新类别 (16-20) 使用 step0 最佳模型的平均权重
                for i in range(old_num_classes, new_num_classes):
                    new_weight[i] = avg_weight[0]
                
                # 更新模型权重
                conv_seg.weight.data = new_weight
                
                if logger:
                    logger.info(f"✓ decode_head 分类头权重已复制")
                    logger.info(f"   - 旧类别 (0-{old_num_classes-1}): 使用Step0对应类别权重")
                    logger.info(f"   - 新类别 ({old_num_classes}-{new_num_classes-1}): 使用Step0所有类别平均权重初始化")
                    logger.info(f"   - 权重形状: {old_weight.shape} -> {new_weight.shape}")
        
        # 处理 bias（如果有）
        if hasattr(decode_head, 'conv_seg') and decode_head.conv_seg.bias is not None:
            old_bias_key = 'decode_head.conv_seg.bias'
            if old_bias_key in step0_state_dict:
                old_bias = step0_state_dict[old_bias_key]  # [old_num_classes]
                new_bias = torch.zeros(new_num_classes, device=old_bias.device, dtype=old_bias.dtype)
                new_bias[:old_num_classes] = old_bias
                
                # 新类别使用 step0 最佳模型的平均bias
                avg_bias = old_bias.mean()
                new_bias[old_num_classes:] = avg_bias
                
                decode_head.conv_seg.bias.data = new_bias
    
    # 扩展 auxiliary_head 分类头
    if hasattr(model, 'auxiliary_head') and model.auxiliary_head is not None:
        auxiliary_head = model.auxiliary_head
        
        if hasattr(auxiliary_head, 'conv_seg'):
            conv_seg = auxiliary_head.conv_seg
            
            old_weight_key = 'auxiliary_head.conv_seg.weight'
            if old_weight_key in step0_state_dict:
                old_weight = step0_state_dict[old_weight_key]
                
                channels = old_weight.shape[1]
                new_weight = torch.zeros(new_num_classes, channels, 1, 1, device=old_weight.device, dtype=old_weight.dtype)
                
                # 复制旧类别权重
                new_weight[:old_num_classes] = old_weight
                
                # 新类别使用 step0 最佳模型的平均权重
                avg_weight = old_weight.mean(dim=0, keepdim=True)
                for i in range(old_num_classes, new_num_classes):
                    new_weight[i] = avg_weight[0]
                
                conv_seg.weight.data = new_weight
                
                if logger:
                    logger.info(f"✓ auxiliary_head 分类头权重已复制")
                    logger.info(f"   - 旧类别 (0-{old_num_classes-1}): 使用Step0对应类别权重")
                    logger.info(f"   - 新类别 ({old_num_classes}-{new_num_classes-1}): 使用Step0所有类别平均权重初始化")
                    logger.info(f"   - 权重形状: {old_weight.shape} -> {new_weight.shape}")
        
        # 处理 bias（如果有）
        if hasattr(auxiliary_head, 'conv_seg') and auxiliary_head.conv_seg.bias is not None:
            old_bias_key = 'auxiliary_head.conv_seg.bias'
            if old_bias_key in step0_state_dict:
                old_bias = step0_state_dict[old_bias_key]
                new_bias = torch.zeros(new_num_classes, device=old_bias.device, dtype=old_bias.dtype)
                new_bias[:old_num_classes] = old_bias
                
                # 新类别使用 step0 最佳模型的平均bias
                avg_bias = old_bias.mean()
                new_bias[old_num_classes:] = avg_bias
                
                auxiliary_head.conv_seg.bias.data = new_bias
    
    if logger:
        logger.info(f"✓ 分类头扩展完成")
        logger.info(f"   - 保留旧类别 (0-{old_num_classes-1}) 的权重")
        logger.info(f"   - 新类别 ({old_num_classes}-{new_num_classes-1}) 使用Step0所有类别平均权重初始化")


# ==================== 加载 Step0 模型（用于生成旧类别预测） ====================
def load_step0_model_for_predictions(step0_checkpoint_path, device, logger=None):
    """
    加载 Step0 模型用于生成旧类别的预测
    
    Args:
        step0_checkpoint_path: Step0 检查点路径
        device: 设备
        logger: 日志记录器
    
    Returns:
        step0_model: Step0 模型（16类）
    """
    if not os.path.exists(step0_checkpoint_path):
        error_msg = f"✗ Step0 检查点文件不存在: {step0_checkpoint_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if logger:
        logger.info(f"\n加载 Step0 模型用于生成旧类别预测...")
        logger.info(f"✓ Step0 检查点: {step0_checkpoint_path}")
    
    # 加载 mmsegmentation 配置文件（与 Step0 保持一致，使用 Tiny 配置）
    config_file = os.path.join(
        mmseg_dir, 
        'configs/swin/upernet_swin_tiny_patch4_window7_512x512_40k_voc12aug.py'
    )
    
    # 加载配置
    cfg = Config.fromfile(config_file)
    
    # Step0 有 16 类
    cfg.model.decode_head.num_classes = 16
    cfg.model.auxiliary_head.num_classes = 16
    
    # 不加载预训练模型（我们将从 checkpoint 加载）
    cfg.model.pretrained = None
    
    # 构建模型
    step0_model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    step0_model = step0_model.to(device)
    step0_model.eval()
    
    # 加载权重
    checkpoint = torch.load(step0_checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        step0_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        step0_model.load_state_dict(checkpoint)
    
    # 冻结模型参数
    for param in step0_model.parameters():
        param.requires_grad = False
    
    if logger:
        logger.info(f"✓ Step0 模型加载成功（16类，用于生成旧类别预测）")
    
    return step0_model


# ==================== 加载 Step0 模型权重 ====================
def load_step0_weights(model, step0_checkpoint_path, device, logger=None):
    """
    加载 Step0 模型权重到新模型（除了分类头）
    
    Args:
        model: 新模型（21类）
        step0_checkpoint_path: Step0 检查点路径
        device: 设备
        logger: 日志记录器
    """
    if not os.path.exists(step0_checkpoint_path):
        error_msg = f"✗ Step0 检查点文件不存在: {step0_checkpoint_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise FileNotFoundError(error_msg)
    
    if logger:
        logger.info(f"\n加载 Step0 最佳模型权重...")
        logger.info(f"✓ Step0 检查点: {step0_checkpoint_path}")
    
    checkpoint = torch.load(step0_checkpoint_path, map_location=device)
    
    # 获取模型状态字典
    if 'model_state_dict' in checkpoint:
        step0_state_dict = checkpoint['model_state_dict']
    else:
        step0_state_dict = checkpoint
    
    # 获取新模型的状态字典
    new_state_dict = model.state_dict()
    
    # 复制权重（排除分类头）
    copied_keys = []
    skipped_keys = []
    
    for key, value in step0_state_dict.items():
        # 跳过分类头相关的键
        if 'decode_head.conv_seg' in key or 'auxiliary_head.conv_seg' in key:
            skipped_keys.append(key)
            continue
        
        # 如果键在新模型中存在且形状匹配，则复制
        if key in new_state_dict:
            if new_state_dict[key].shape == value.shape:
                new_state_dict[key] = value
                copied_keys.append(key)
            else:
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)
    
    # 加载更新后的状态字典
    model.load_state_dict(new_state_dict, strict=False)
    
    if logger:
        logger.info(f"✓ Step0 权重加载成功（backbone 和大部分层）")
        if skipped_keys:
            logger.info(f"  跳过 {len(skipped_keys)} 个不匹配的键（主要是分类头）")
    
    return step0_state_dict


# ==================== 训练函数 ====================
def train_epoch(model, train_loader, optimizer, epoch, max_epochs, max_iters, 
                base_lr, device, config, val_loader=None, val_metrics=None, 
                best_val_iou_list=None, output_dir=None, logger=None, step0_model=None, start_iter=0):
    """
    训练一个epoch，支持迭代级别的验证和模型保存
    
    使用混合标签策略：
    - 旧类别（1-15）：使用 Step0 模型的预测
    - 新类别（16-20）和背景类（0）：使用真实标签
    """
    model.train()
    if step0_model is not None:
        step0_model.eval()  # Step0 模型始终处于评估模式
    
    total_loss = 0.0
    iter_num = 0
    
    # 使用列表来传递 best_val_iou，以便在函数内部修改
    if best_val_iou_list is None:
        best_val_iou_list = [0.0]
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', leave=False)
    
    for batch_idx, (images, labels, l1h) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long()
        
        # 计算当前迭代数（考虑恢复训练的起始迭代数）
        current_iter = start_iter + epoch * len(train_loader) + batch_idx
        
        # 更新学习率
        lr = update_learning_rate_poly(
            optimizer, current_iter, max_iters, base_lr,
            power=config.power, min_lr=config.min_lr,
            warmup_iters=config.warmup_iters, warmup_ratio=config.warmup_ratio
        )
        
        # 准备数据格式（mmsegmentation 格式）
        img_metas = [{
            'img_shape': images.shape[2:],
            'ori_shape': images.shape[2:],
            'pad_shape': images.shape[2:],
            'scale_factor': 1.0,
            'flip': False
        } for _ in range(images.size(0))]
        
        # ==================== 生成混合标签 ====================
        # 对于旧类别（1-15），使用 Step0 模型的预测
        # 对于新类别（16-20）和背景类（0），使用真实标签
        
        if step0_model is not None:
            # 使用 Step0 模型生成旧类别的预测
            with torch.no_grad():
                step0_logits = step0_model.encode_decode(images, img_metas)  # [B, 16, H, W]
                step0_predictions = step0_logits.argmax(dim=1)  # [B, H, W] - 旧类别预测（0-15）
            
            # 创建混合标签：初始化为真实标签
            mixed_labels = labels.clone()  # [B, H, W] - 初始化为真实标签（包含所有21类）
            
            # 对于旧类别（1-15），使用 Step0 模型的预测
            # Step0 模型的类别映射：
            #   - Step0 类别 0 (背景) -> 新模型类别 0 (背景) - 但这里我们用真实标签
            #   - Step0 类别 1-15 (旧类别) -> 新模型类别 1-15 (旧类别) - 使用 Step0 预测
            
            # 找到 Step0 模型预测为旧类别（1-15）的像素
            old_class_mask = (step0_predictions >= 1) & (step0_predictions <= 15)  # [B, H, W]
            
            # 将旧类别区域的标签替换为 Step0 模型的预测
            # Step0 模型的类别 1-15 直接对应新模型的类别 1-15
            mixed_labels[old_class_mask] = step0_predictions[old_class_mask]
            
            # 对于新类别（16-20）和背景类（0），保持使用真实标签
            # 这已经在 mixed_labels 初始化时设置好了，不需要额外处理
            
            # 确保标签格式为 [N, 1, H, W]（mmsegmentation 期望的格式）
            if mixed_labels.dim() == 3:  # [N, H, W]
                mixed_labels = mixed_labels.unsqueeze(1)  # [N, 1, H, W]
        else:
            # 如果没有 Step0 模型，使用真实标签
            if labels.dim() == 3:  # [N, H, W]
                mixed_labels = labels.unsqueeze(1)  # [N, 1, H, W]
            else:
                mixed_labels = labels
        
        # 前向传播（训练模式）
        losses = model.forward_train(images, img_metas, mixed_labels)
        
        # 计算总损失
        loss = sum(losses.values())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        iter_num += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{lr:.6f}'
        })
        
        # 迭代级别的验证（如果启用）
        if (config.eval_interval_iters > 0 and 
            val_loader is not None and 
            val_metrics is not None and
            (current_iter + 1) % config.eval_interval_iters == 0):
            
            # 进行验证
            val_loss, val_score = validate(model, val_loader, val_metrics, device)
            
            current_epoch = current_iter / len(train_loader)
            current_iou = val_score['Mean IoU']
            previous_best_iou = best_val_iou_list[0]
            
            if logger:
                logger.info(f"\n[迭代 {current_iter+1}] 验证结果 (Epoch {current_epoch:.2f}):")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Mean IoU: {current_iou:.4f}")
            
            # 检查是否是最佳模型
            is_best_iou = current_iou > previous_best_iou
            
            if is_best_iou:
                best_val_iou_list[0] = current_iou
                
                # 保存检查点
                if output_dir is not None:
                    checkpoint_path = os.path.join(
                        output_dir, 
                        f'checkpoint_iter_{current_iter+1}_best_{current_iou:.4f}.pth'
                    )
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, checkpoint_path,
                        is_best=True, best_val_iou=current_iou, current_iter=current_iter+1
                    )
                    
                    if logger:
                        logger.info(
                            f"✓ 保存模型 (迭代 {current_iter+1}, "
                            f"当前Mean IoU: {current_iou:.4f} > 历史最高: {previous_best_iou:.4f})"
                        )
            else:
                if logger:
                    logger.info(
                        f"  跳过保存 (当前Mean IoU: {current_iou:.4f} <= 历史最高: {previous_best_iou:.4f})"
                    )
    
    avg_loss = total_loss / iter_num if iter_num > 0 else 0.0
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath, is_best=False, best_val_iou=None, current_iter=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if best_val_iou is not None:
        checkpoint['best_val_iou'] = best_val_iou
    
    # 保存当前迭代数（用于学习率调度器）
    if current_iter is not None:
        checkpoint['current_iter'] = current_iter
    
    if is_best:
        filepath = filepath.replace('.pth', '_best.pth')
    
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(model, optimizer, checkpoint_path, device, logger=None):
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('loss', float('inf'))
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    current_iter = checkpoint.get('current_iter', None)
    
    if logger:
        logger.info(f"✓ 检查点加载成功")
        logger.info(f"  恢复epoch: {start_epoch}")
        logger.info(f"  检查点损失: {best_val_loss:.4f}")
        if 'best_val_iou' in checkpoint:
            logger.info(f"  检查点IoU: {best_val_iou:.4f}")
        if current_iter is not None:
            logger.info(f"  恢复迭代数: {current_iter}")
    
    return start_epoch, best_val_loss, best_val_iou, current_iter


# ==================== 验证函数 ====================
def validate(model, val_loader, metrics, device):
    """验证函数"""
    model.eval()
    metrics.reset()
    
    total_loss = 0.0
    iter_num = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
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
            
            # 确保标签格式为 [N, 1, H, W]
            labels_for_loss = labels.unsqueeze(1) if labels.dim() == 3 else labels
            
            # 前向传播（测试模式）
            seg_logits = model.encode_decode(images, img_metas)
            
            # 获取预测结果（argmax）
            _, predictions = seg_logits.max(dim=1)  # [B, H, W]
            
            # 计算损失（用于监控）
            losses = model.forward_train(images, img_metas, labels_for_loss)
            loss = sum(losses.values())
            
            total_loss += loss.item()
            iter_num += 1
            
            # 转换为numpy并更新评估指标
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            metrics.update(labels_np, predictions_np)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
    
    # 计算评估得分
    score = metrics.get_results()
    
    avg_loss = total_loss / iter_num if iter_num > 0 else 0.0
    
    return avg_loss, score


# ==================== 主函数 ====================
def main(config=None):
    """主函数：加载 Step0 模型，扩展分类头，使用 Step1 样本训练"""
    print("=" * 70)
    print("mmsegmentation Swin-Large Step1 - 加载 Step0 最佳模型并训练")
    print("=" * 70)
    
    if config is None:
        config = TrainingConfig()
    
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
    log_file = os.path.join(config.output_dir, 'training.log')
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
    logger.info("训练配置:")
    logger.info(f"  模型: Swin-Tiny (mmsegmentation) - 与 Step0 保持一致")
    logger.info(f"  任务: {config.task} Step {config.step}")
    logger.info(f"  最大epoch数: {config.max_epochs}")
    logger.info(f"  基础学习率: {config.base_lr}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Step0 检查点: {config.step0_checkpoint}")
    logger.info(f"  设备: {config.device}")
    logger.info(f"  输出目录: {config.output_dir}")
    logger.info(f"  验证频率: 每 {config.eval_interval_iters} 次迭代验证一次")
    logger.info(f"  Epoch验证: 每 {config.eval_interval} 个epoch验证一次")
    logger.info("=" * 70)
    
    # ==================== 加载数据集 ====================
    logger.info("\n加载数据集...")
    
    dataset_config = VOCDatasetConfig()
    dataset_config.task = config.task
    dataset_config.step = config.step
    dataset_config.batch_size = config.batch_size
    dataset_config.num_workers = config.num_workers
    dataset_config.crop_size = config.crop_size
    dataset_config.crop_size_val = config.crop_size_val
    dataset_config.data_root = config.data_root
    
    try:
        train_loader, val_loader, test_loader, n_classes = create_dataloaders(
            dataset_config,
            distributed=False
        )
        logger.info(f"✓ 数据集加载成功")
        logger.info(f"  训练集: {len(train_loader.dataset)} 样本")
        logger.info(f"  验证集: {len(val_loader.dataset)} 样本")
        logger.info(f"  测试集: {len(test_loader.dataset)} 样本")
        logger.info(f"  类别数: {n_classes}")
        logger.info(f"  图像尺寸: {config.crop_size}x{config.crop_size_val}")
    except Exception as e:
        logger.error(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 加载模型 ====================
    logger.info("\n加载模型...")
    
    # 加载 mmsegmentation 配置文件（与 Step0 保持一致，使用 Tiny 配置）
    config_file = os.path.join(
        mmseg_dir, 
        'configs/swin/upernet_swin_tiny_patch4_window7_512x512_40k_voc12aug.py'
    )
    
    if not os.path.exists(config_file):
        logger.error(f"✗ 配置文件不存在: {config_file}")
        return
    
    logger.info(f"✓ 配置文件: {config_file}")
    logger.info(f"  使用与 Step0 相同的配置（Swin-Tiny + UPerNet）")
    
    # 加载配置
    cfg = Config.fromfile(config_file)
    
    # 更新类别数（Step1 有 21 类）
    cfg.model.decode_head.num_classes = n_classes
    cfg.model.auxiliary_head.num_classes = n_classes
    
    # 不加载预训练模型（我们将从 Step0 加载）
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
    
    # ==================== 加载 Step0 权重并扩展分类头 ====================
    try:
        # 加载 Step0 权重（除了分类头）
        step0_state_dict = load_step0_weights(
            model, config.step0_checkpoint, config.device, logger
        )
        
        # 扩展分类头并复制权重
        expand_classification_head(
            model, step0_state_dict, 
            old_num_classes=16, new_num_classes=21, 
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"✗ 加载 Step0 权重失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 加载 Step0 模型用于生成旧类别预测 ====================
    step0_model = None
    try:
        step0_model = load_step0_model_for_predictions(
            config.step0_checkpoint, config.device, logger
        )
        logger.info("✓ Step0 模型已加载，将用于生成旧类别的预测标签")
    except Exception as e:
        logger.warning(f"⚠ 加载 Step0 模型失败: {e}")
        logger.warning("  将使用真实标签进行训练（不使用混合标签策略）")
        import traceback
        traceback.print_exc()
    
    # ==================== 设置优化器 ====================
    logger.info("\n设置优化器...")
    
    # 从配置文件构建优化器（支持 paramwise_cfg，与 Step0 一致）
    if hasattr(cfg, 'optimizer') and cfg.optimizer is not None:
        # 使用配置文件中的优化器设置（包含 paramwise_cfg，与 Step0 一致）
        optimizer = build_optimizer(model, cfg.optimizer)
        logger.info(f"✓ 优化器: {cfg.optimizer.type}")
        logger.info(f"  学习率: {cfg.optimizer.lr}")
        logger.info(f"  权重衰减: {cfg.optimizer.weight_decay}")
        if hasattr(cfg.optimizer, 'paramwise_cfg'):
            logger.info(f"  参数特定配置: {cfg.optimizer.paramwise_cfg}")
    else:
        # 回退到默认配置
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        logger.info(f"✓ 优化器: {config.optimizer_type}")
        logger.info(f"  学习率: {config.base_lr}")
        logger.info(f"  权重衰减: {config.weight_decay}")
    
    # ==================== 创建评估指标对象 ====================
    logger.info("\n创建评估指标对象...")
    val_metrics = StreamSegMetrics(n_classes=n_classes)
    test_metrics = StreamSegMetrics(n_classes=n_classes)
    logger.info(f"✓ 评估指标对象创建成功 (类别数: {n_classes})")
    
    # ==================== 训练循环 ====================
    logger.info("\n" + "=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)
    
    max_iters = config.max_epochs * len(train_loader)
    best_val_loss = float('inf')
    best_val_iou = 0.0
    start_epoch = 0
    start_iter = 0  # 起始迭代数（用于学习率调度器）
    
    # 加载检查点（如果指定）
    if config.resume is not None:
        if os.path.isabs(config.resume):
            checkpoint_path = config.resume
        else:
            checkpoint_path = os.path.join(current_dir, config.resume)
        
        if os.path.exists(checkpoint_path):
            logger.info(f"\n从检查点恢复训练: {checkpoint_path}")
            try:
                start_epoch, best_val_loss, best_val_iou, restored_iter = load_checkpoint(
                    model, optimizer, checkpoint_path, config.device, logger
                )
                # 如果检查点中有保存的迭代数，使用它；否则根据epoch计算
                if restored_iter is not None:
                    start_iter = restored_iter
                    logger.info(f"✓ 恢复迭代数: {start_iter}")
                else:
                    # 如果没有保存迭代数，根据epoch估算（可能不准确，但比从头开始好）
                    start_iter = start_epoch * len(train_loader)
                    logger.info(f"⚠ 检查点中没有迭代数，根据epoch估算: {start_iter}")
                
                # 如果扩展训练（max_epochs增加了），需要调整max_iters
                # 但保持学习率调度器从正确的位置继续
                original_max_iters = start_epoch * len(train_loader) if start_epoch > 0 else max_iters
                logger.info(f"  原始训练计划迭代数: {original_max_iters}")
                logger.info(f"  新训练计划总迭代数: {max_iters}")
                logger.info(f"  将从迭代 {start_iter} 继续训练到 {max_iters}")
                logger.info(f"✓ 将从 epoch {start_epoch} 继续训练到 epoch {config.max_epochs}")
            except Exception as e:
                logger.error(f"✗ 加载检查点失败: {e}")
                logger.error("  将从头开始训练")
                start_epoch = 0
                start_iter = 0
        else:
            logger.warning(f"⚠ 检查点文件不存在: {checkpoint_path}")
            logger.warning("  将从头开始训练")
    
    for epoch in range(start_epoch, config.max_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.max_epochs} - 开始训练")
        logger.info(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练（传入验证相关参数以支持迭代级别验证和模型保存）
        # 使用列表传递 best_val_iou，以便在 train_epoch 内部修改
        best_val_iou_list = [best_val_iou]
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, config.max_epochs, max_iters,
            config.base_lr, config.device, config,
            val_loader=val_loader if config.eval_interval_iters > 0 else None,
            val_metrics=val_metrics if config.eval_interval_iters > 0 else None,
            best_val_iou_list=best_val_iou_list,
            output_dir=config.output_dir if config.eval_interval_iters > 0 else None,
            logger=logger,
            step0_model=step0_model,  # 传入 Step0 模型用于生成旧类别预测
            start_iter=start_iter  # 传递起始迭代数
        )
        # 更新 best_val_iou（可能在迭代级别验证中被更新）
        best_val_iou = best_val_iou_list[0]
        
        logger.info(f"Epoch {epoch+1}/{config.max_epochs} - Train Loss: {train_loss:.4f}")
        
        # 验证
        if (epoch + 1) % config.eval_interval == 0:
            val_loss, val_score = validate(
                model, val_loader, val_metrics, config.device
            )
            
            logger.info(f"Epoch {epoch+1}/{config.max_epochs} - Val Loss: {val_loss:.4f}")
            
            logger.info("\n" + "-" * 70)
            logger.info("验证评估指标:")
            logger.info(f"  Overall Accuracy: {val_score['Overall Acc']:.4f}")
            logger.info(f"  Mean Accuracy: {val_score['Mean Acc']:.4f}")
            logger.info(f"  Mean Precision: {val_score['Mean Prec']:.4f}")
            logger.info(f"  Mean IoU: {val_score['Mean IoU']:.4f}")
            logger.info("-" * 70)
            
            if (epoch + 1) % (config.eval_interval * 5) == 0 or epoch == config.max_epochs - 1:
                logger.info("\n各类别IoU:")
                for class_id, iou in val_score['Class IoU'].items():
                    if iou != "X":
                        logger.info(f"  类别 {class_id}: {iou:.4f}")
            
            checkpoint_path = os.path.join(config.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            
            current_iou = val_score['Mean IoU']
            previous_best_iou = best_val_iou
            is_best_iou = current_iou > best_val_iou
            
            if is_best_iou:
                best_val_iou = current_iou
                # 计算当前迭代数（考虑恢复训练的起始迭代数）
                current_iter = start_iter + (epoch + 1) * len(train_loader)
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, 
                              is_best=True, best_val_iou=best_val_iou, current_iter=current_iter)
                logger.info(f"✓ 保存模型 (当前Mean IoU: {current_iou:.4f} > 历史最高: {previous_best_iou:.4f})")
            else:
                logger.info(f"  跳过保存 (当前Mean IoU: {current_iou:.4f} <= 历史最高: {previous_best_iou:.4f})")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    
    logger.info("\n" + "=" * 70)
    logger.info("训练完成!")
    logger.info("=" * 70)
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"最佳验证IoU: {best_val_iou:.4f}")
    logger.info(f"模型保存在: {config.output_dir}")
    
    # ==================== 最终测试评估 ====================
    logger.info("\n" + "=" * 70)
    logger.info("在测试集上进行最终评估...")
    logger.info("=" * 70)
    
    test_score = test(model, test_loader, test_metrics, config.device)
    
    logger.info("\n测试集评估结果:")
    logger.info(f"  Overall Accuracy: {test_score['Overall Acc']:.4f}")
    logger.info(f"  Mean Accuracy: {test_score['Mean Acc']:.4f}")
    logger.info(f"  Mean Precision: {test_score['Mean Prec']:.4f}")
    logger.info(f"  Mean IoU: {test_score['Mean IoU']:.4f}")
    
    logger.info("\n测试集各类别IoU:")
    for class_id, iou in test_score['Class IoU'].items():
        if iou != "X":
            logger.info(f"  类别 {class_id}: {iou:.4f}")


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
            _, predictions = seg_logits.max(dim=1)  # [B, H, W]
            
            # 转换为numpy并更新评估指标
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            metrics.update(labels_np, predictions_np)
    
    # 计算评估得分
    score = metrics.get_results()
    
    return score


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='训练 Step1 模型（基于 Step0 最佳模型）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 从头开始训练:
     python train_mmseg_swin_large_voc_15_5_step1.py
  
  2. 从最佳模型检查点恢复训练（扩展训练到50个epoch）:
     python train_mmseg_swin_large_voc_15_5_step1.py \\
         --resume outputs/mmseg_swin_tiny_voc_15-5_step1/checkpoint_epoch_12_best.pth
  
  3. 使用绝对路径:
     python train_mmseg_swin_large_voc_15_5_step1.py \\
         --resume /home/shl/pps/WILSS/outputs/mmseg_swin_tiny_voc_15-5_step1/checkpoint_epoch_12_best.pth

注意:
  - 恢复训练时会自动保持学习率调度器的状态（基于迭代数）
  - 如果max_epochs增加了（如从12到50），训练会从检查点的epoch继续到新的max_epochs
  - 学习率会根据恢复的迭代数继续正确衰减
        """
    )
    parser.add_argument('--step0-checkpoint', type=str, 
                        default='/home/shl/pps/WILSS/outputs/mmseg_swin_tiny_voc_15-5_step0/checkpoint_iter_36500_best_best.pth',
                        help='Step0 检查点路径（16类训练的最佳模型）')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径（相对于输出目录或绝对路径）。'
                             '建议使用最佳模型检查点（如 checkpoint_epoch_*_best.pth）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='恢复训练的检查点路径（别名，与 --resume 相同）')
    
    args = parser.parse_args()
    
    # 创建配置并设置 step0_checkpoint
    config = TrainingConfig()
    if args.step0_checkpoint is not None:
        config.step0_checkpoint = args.step0_checkpoint
    
    # 如果提供了 --checkpoint，使用它；否则使用 --resume
    resume_path = args.checkpoint if args.checkpoint is not None else args.resume
    if resume_path is not None:
        config.resume = resume_path
    
    main(config)

