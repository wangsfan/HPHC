"""
使用 mmsegmentation 的 Swin-Tiny 模型训练 VOC 15-5 Step0 分阶段分割任务
结合 mmsegmentation 的模型架构和 WILSON 的分阶段数据加载
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
        # 基本训练参数
        self.max_epochs = 50  # 可以设置为30或更多，支持从检查点恢复继续训练
        self.base_lr = 0.00006  # Swin-Tiny 推荐学习率
        self.batch_size = 8  # Swin-Tiny 使用 batch_size=8
        self.num_workers = 4
        self.eval_interval = 1  # 每几个epoch验证一次（epoch级别的验证）
        self.eval_interval_iters = 500  # 每多少次迭代验证一次（迭代级别的验证，与mmsegmentation一致）
        
        # 数据集配置
        self.task = '15-5'
        self.step = 0
        self.data_root = os.path.join(current_dir, 'WILSON', 'data')
        self.crop_size = 512  # mmsegmentation 标准尺寸
        self.crop_size_val = 512
        
        # 模型配置
        self.model_size = 'tiny'  # 'tiny', 'small', 'base', 'large'
        self.pretrained_model = os.path.join(
            mmseg_dir, 
            'checkpoints/swin_tiny_patch4_window7_224.pth'
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


# ==================== 训练函数 ====================
def train_epoch(model, train_loader, optimizer, epoch, max_epochs, max_iters, 
                base_lr, device, config, val_loader=None, val_metrics=None, 
                best_val_iou_list=None, output_dir=None, logger=None, start_iter=0):
    """训练一个epoch，支持迭代级别的验证和模型保存"""
    model.train()
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
        
        # 确保标签格式为 [N, 1, H, W]（mmsegmentation 期望的格式）
        if labels.dim() == 3:  # [N, H, W]
            labels = labels.unsqueeze(1)  # [N, 1, H, W]
        
        # 前向传播（训练模式）
        losses = model.forward_train(images, img_metas, labels)
        
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
                        f'checkpoint_iter_{current_iter+1}_best.pth'
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


# ==================== 主训练函数 ====================
def main(config=None):
    """主训练函数"""
    print("=" * 70)
    print("mmsegmentation Swin-Tiny 训练 - VOC 15-5 Step0")
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
    logger.info(f"  模型: Swin-Tiny (mmsegmentation)")
    logger.info(f"  任务: {config.task} Step {config.step}")
    logger.info(f"  最大epoch数: {config.max_epochs}")
    logger.info(f"  基础学习率: {config.base_lr}")
    logger.info(f"  Batch Size: {config.batch_size}")
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
    
    # 加载 mmsegmentation 配置文件
    config_file = os.path.join(
        mmseg_dir, 
        'configs/swin/upernet_swin_tiny_patch4_window7_512x512_40k_voc12aug.py'
    )
    
    if not os.path.exists(config_file):
        logger.error(f"✗ 配置文件不存在: {config_file}")
        return
    
    logger.info(f"✓ 配置文件: {config_file}")
    
    # 加载配置
    cfg = Config.fromfile(config_file)
    
    # 更新类别数
    cfg.model.decode_head.num_classes = n_classes
    cfg.model.auxiliary_head.num_classes = n_classes
    
    # 更新预训练模型路径
    if os.path.exists(config.pretrained_model):
        cfg.model.pretrained = config.pretrained_model
        logger.info(f"✓ 预训练模型: {config.pretrained_model}")
    else:
        logger.warning(f"⚠ 预训练模型不存在: {config.pretrained_model}")
        logger.warning("  将使用随机初始化的权重")
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
    
    # ==================== 设置优化器 ====================
    logger.info("\n设置优化器...")
    
    # 从配置文件构建优化器（支持 paramwise_cfg）
    if hasattr(cfg, 'optimizer') and cfg.optimizer is not None:
        # 使用配置文件中的优化器设置（包含 paramwise_cfg）
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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='训练 mmsegmentation Swin-Tiny 模型 (VOC 15-5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 从头开始训练:
     python train_mmseg_swin_large_voc_15_5_step0_saved.py
  
  2. 从最佳模型检查点恢复训练（扩展训练到30个epoch）:
     python train_mmseg_swin_large_voc_15_5_step0_saved.py \\
         --resume outputs/mmseg_swin_tiny_voc_15-5_step0/checkpoint_epoch_10_best.pth
  
  3. 使用绝对路径:
     python train_mmseg_swin_large_voc_15_5_step0_saved.py \\
         --resume /home/shl/pps/WILSS/outputs/mmseg_swin_tiny_voc_15-5_step0/checkpoint_epoch_10_best.pth

注意:
  - 恢复训练时会自动保持学习率调度器的状态（基于迭代数）
  - 如果max_epochs增加了（如从10到30），训练会从检查点的epoch继续到新的max_epochs
  - 学习率会根据恢复的迭代数继续正确衰减
        """
    )
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径（相对于输出目录或绝对路径）。'
                             '建议使用最佳模型检查点（如 checkpoint_epoch_*_best.pth）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='恢复训练的检查点路径（别名，与 --resume 相同）')
    
    args = parser.parse_args()
    
    # 如果提供了 --checkpoint，使用它；否则使用 --resume
    resume_path = args.checkpoint if args.checkpoint is not None else args.resume
    
    # 创建配置并设置 resume
    config = TrainingConfig()
    if resume_path is not None:
        config.resume = resume_path
    
    main(config)

