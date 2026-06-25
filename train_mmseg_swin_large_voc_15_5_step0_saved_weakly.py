"""
使用 mmsegmentation 的 Swin-Large + UPerNet 模型训练 VOC 15-5 Step0
使用 SIPE 生成的伪标签进行弱监督训练
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

# 导入 SwanLab 用于实验跟踪
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("警告: SwanLab 未安装，将跳过实验跟踪。")
    print("      训练日志仍会正常记录到文件。")

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
from data_loader import create_dataloaders, get_task_dict, get_task_labels
from load_voc_dataset import VOCDatasetConfig

# 导入评估指标
from metrics import StreamSegMetrics


# ==================== 训练配置 ====================
class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        # 基本训练参数
        self.max_epochs = 20  # 使用伪标签训练，可以少一些epoch
        self.base_lr = 0.00006  # Swin-Large 推荐学习率
        self.batch_size = 1  # Swin-Large 模型较大，使用 batch_size=1
        self.num_workers = 4
        self.eval_interval = 1  # 每几个epoch验证一次
        self.eval_interval_iters = 1000  # 每多少次迭代验证一次
        
        # 数据集配置
        self.task = '15-5'
        self.step = 0
        self.data_root = os.path.join(current_dir, 'WILSON', 'data')
        self.crop_size = 512
        self.crop_size_val = 512
        
        # 模型配置
        self.model_size = 'large'
        self.pretrained_model = os.path.join(
            mmseg_dir, 
            'checkpoints/swin_large_patch4_window7_224_22k.pth'
        )
        
        # 优化器配置
        self.optimizer_type = 'AdamW'
        self.weight_decay = 0.01
        self.betas = (0.9, 0.999)
        
        # 学习率调度器配置
        self.lr_scheduler = 'poly'
        self.power = 1.0
        self.min_lr = 0.0
        self.warmup_iters = 1500
        self.warmup_ratio = 1e-6
        
        # 输出目录
        self.output_dir = os.path.join(
            current_dir, 
            'outputs', 
            f'mmseg_swin_{self.model_size}_voc_{self.task}_step{self.step}_weakly_sipe'
        )
        
        # 恢复训练
        self.resume = None
        
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
    """多项式学习率调度"""
    if current_iter < warmup_iters:
        lr = base_lr * (warmup_ratio + (1 - warmup_ratio) * current_iter / warmup_iters)
    else:
        lr = base_lr * ((1 - current_iter / max_iters) ** power)
        lr = max(lr, min_lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


# ==================== 训练函数 ====================
def train_epoch(model, train_loader, optimizer, epoch, max_epochs, max_iters, 
                base_lr, device, config, val_loader=None, val_metrics=None, 
                best_val_iou_list=None, output_dir=None, logger=None, swanlab_run=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    iter_num = 0
    
    if best_val_iou_list is None:
        best_val_iou_list = [0.0]
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}', leave=False)
    
    for batch_idx, (images, labels, l1h) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long()
        
        current_iter = epoch * len(train_loader) + batch_idx
        
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
        
        # 确保标签格式为 [N, 1, H, W]
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        
        # 前向传播（训练模式）
        losses = model.forward_train(images, img_metas, labels)
        
        # 计算总损失
        loss = sum(losses.values())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        iter_num += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{lr:.6f}'
        })
        
        # 记录训练指标到 SwanLab（每100个iteration记录一次，避免记录太频繁）
        if swanlab_run is not None and (current_iter + 1) % 100 == 0:
            log_dict = {
                'train/loss': loss.item(),
                'train/learning_rate': lr,
                'train/iteration': current_iter + 1,
                'train/epoch': epoch + 1
            }
            # 记录各个损失分量
            for loss_name, loss_value in losses.items():
                log_dict[f'train/loss_{loss_name}'] = loss_value.item()
            swanlab.log(log_dict)
        
        # 迭代级别的验证
        if (config.eval_interval_iters > 0 and 
            val_loader is not None and 
            val_metrics is not None and
            (current_iter + 1) % config.eval_interval_iters == 0):
            
            val_loss, val_score = validate(model, val_loader, val_metrics, device, swanlab_run=swanlab_run)
            
            current_epoch = current_iter / len(train_loader)
            current_iou = val_score['Mean IoU']
            previous_best_iou = best_val_iou_list[0]
            
            if logger:
                logger.info(f"\n[迭代 {current_iter+1}] 验证结果 (Epoch {current_epoch:.2f}):")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Mean IoU: {current_iou:.4f}")
            
            # 记录验证指标到 SwanLab
            if swanlab_run is not None:
                val_log_dict = {
                    'val/loss': val_loss,
                    'val/miou': current_iou,
                    'val/overall_acc': val_score['Overall Acc'],
                    'val/mean_acc': val_score['Mean Acc'],
                    'val/mean_prec': val_score['Mean Prec'],
                    'val/iteration': current_iter + 1,
                    'val/epoch': current_epoch
                }
                swanlab.log(val_log_dict)
            
            is_best_iou = current_iou > previous_best_iou
            
            if is_best_iou:
                best_val_iou_list[0] = current_iou
                
                if output_dir is not None:
                    checkpoint_path = os.path.join(
                        output_dir, 
                        f'checkpoint_iter_{current_iter+1}_best.pth'
                    )
                    try:
                        save_checkpoint(
                            model, optimizer, epoch, val_loss, checkpoint_path,
                            is_best=True, best_val_iou=current_iou, logger=logger
                        )
                    except Exception as e:
                        if logger:
                            logger.warning(f"保存检查点失败，但训练继续: {e}")
                    
                    if logger:
                        logger.info(
                            f"✓ 保存模型 (迭代 {current_iter+1}, "
                            f"当前Mean IoU: {current_iou:.4f} > 历史最高: {previous_best_iou:.4f})"
                        )
    
    avg_loss = total_loss / iter_num if iter_num > 0 else 0.0
    return avg_loss


def validate(model, val_loader, metrics, device, swanlab_run=None):
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
            
            img_metas = [{
                'img_shape': images.shape[2:],
                'ori_shape': images.shape[2:],
                'pad_shape': images.shape[2:],
                'scale_factor': 1.0,
                'flip': False
            } for _ in range(images.size(0))]
            
            labels_for_loss = labels.unsqueeze(1) if labels.dim() == 3 else labels
            
            seg_logits = model.encode_decode(images, img_metas)
            _, predictions = seg_logits.max(dim=1)
            
            losses = model.forward_train(images, img_metas, labels_for_loss)
            loss = sum(losses.values())
            
            total_loss += loss.item()
            iter_num += 1
            
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            metrics.update(labels_np, predictions_np)
    
    score = metrics.get_results()
    avg_loss = total_loss / iter_num if iter_num > 0 else 0.0
    
    return avg_loss, score


def save_checkpoint(model, optimizer, epoch, loss, filepath, is_best=False, best_val_iou=None, logger=None):
    """保存检查点（带异常处理和临时文件）"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        if best_val_iou is not None:
            checkpoint['best_val_iou'] = best_val_iou
        
        if is_best:
            filepath = filepath.replace('.pth', '_best.pth')
        
        # 使用临时文件保存，然后原子性移动，避免保存过程中出错导致文件损坏
        temp_filepath = filepath + '.tmp'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存到临时文件
        torch.save(checkpoint, temp_filepath)
        
        # 原子性移动到最终位置
        import shutil
        shutil.move(temp_filepath, filepath)
        
        if logger:
            logger.debug(f"检查点已保存: {filepath}")
        
        return filepath
    except Exception as e:
        if logger:
            logger.error(f"保存检查点失败: {e}")
            import traceback
            traceback.print_exc()
        # 清理临时文件
        temp_filepath = filepath + '.tmp'
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass
        raise


# ==================== 主训练函数 ====================
def main(config=None):
    """主训练函数"""
    print("=" * 70)
    print("mmsegmentation Swin-Large + UPerNet 训练 - VOC 15-5 Step0")
    print("使用 SIPE 伪标签进行弱监督训练")
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
    os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
    
    # 设置日志（如果磁盘空间不足，只使用控制台输出）
    log_file = os.path.join(config.output_dir, 'training.log')
    handlers = [logging.StreamHandler()]  # 总是使用控制台输出
    
    # 尝试添加文件日志，如果失败则只使用控制台
    try:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    except (OSError, IOError) as e:
        print(f"警告: 无法创建日志文件（磁盘空间可能不足），将只使用控制台输出: {e}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # 强制重新配置
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("训练配置:")
    logger.info(f"  模型: Swin-Large + UPerNet (mmsegmentation)")
    logger.info(f"  任务: {config.task} Step {config.step}")
    logger.info(f"  使用 SIPE 伪标签进行弱监督训练")
    logger.info(f"  最大epoch数: {config.max_epochs}")
    logger.info(f"  基础学习率: {config.base_lr}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  设备: {config.device}")
    logger.info(f"  输出目录: {config.output_dir}")
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
    # 使用 SIPE 生成的伪标签
    dataset_config.pseudo = 'sipe'  # 伪标签路径前缀
    dataset_config.weakly = True     # 启用弱监督模式（使用伪标签）
    
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
        logger.info(f"  伪标签路径: {config.data_root}/voc/PseudoLabels/sipe_{config.task}_{config.step}/rw/")
    except Exception as e:
        logger.error(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 加载模型 ====================
    logger.info("\n加载模型...")
    
    config_file = os.path.join(
        mmseg_dir, 
        'configs/swin/upernet_swin_large_patch4_window7_512x512_20k_voc12aug.py'
    )
    
    if not os.path.exists(config_file):
        logger.error(f"✗ 配置文件不存在: {config_file}")
        return
    
    logger.info(f"✓ 配置文件: {config_file}")
    
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
        cfg.model.pretrained = None
    
    # 构建模型
    logger.info("构建模型...")
    try:
        model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        logger.info("✓ 模型构建成功 (Swin-Large + UPerNet)")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  可训练参数数: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 移动到设备
    if config.use_cuda:
        model = model.cuda()
        logger.info("✓ 模型已移动到 GPU")
    else:
        model = model.cpu()
        logger.info("使用 CPU")
    
    # ==================== 设置优化器 ====================
    logger.info("\n设置优化器...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )
    
    logger.info(f"✓ 优化器: {config.optimizer_type}")
    logger.info(f"  学习率: {config.base_lr}")
    logger.info(f"  权重衰减: {config.weight_decay}")
    
    # ==================== 创建评估指标 ====================
    logger.info("\n创建评估指标对象...")
    val_metrics = StreamSegMetrics(n_classes=n_classes)
    test_metrics = StreamSegMetrics(n_classes=n_classes)
    logger.info(f"✓ 评估指标对象创建成功 (类别数: {n_classes})")
    
    # ==================== 初始化 SwanLab ====================
    swanlab_run = None
    if SWANLAB_AVAILABLE:
        try:
            # 尝试初始化 SwanLab，如果磁盘空间不足则跳过
            swanlab_run = swanlab.init(
                project="WILSS-Swin-Large-VOC15-5-Step0-Weakly",
                experiment_name=f"swin_large_voc_{config.task}_step{config.step}_weakly_sipe",
                config={
                    "model": "Swin-Large + UPerNet",
                    "task": config.task,
                    "step": config.step,
                    "max_epochs": config.max_epochs,
                    "batch_size": config.batch_size,
                    "base_lr": config.base_lr,
                    "optimizer": config.optimizer_type,
                    "weight_decay": config.weight_decay,
                    "pseudo_labels": "SIPE",
                    "weakly_supervised": True,
                    "n_classes": n_classes,
                    "crop_size": config.crop_size,
                }
            )
            print("✓ SwanLab 初始化成功")
        except (OSError, IOError) as e:
            if "No space left on device" in str(e):
                print(f"⚠ SwanLab 初始化失败（磁盘空间不足），将跳过实验跟踪")
            else:
                print(f"⚠ SwanLab 初始化失败: {e}")
            swanlab_run = None
        except Exception as e:
            print(f"⚠ SwanLab 初始化失败: {e}")
            swanlab_run = None
    else:
        print("⚠ SwanLab 未安装，将跳过实验跟踪")
    
    # ==================== 训练循环 ====================
    logger.info("\n" + "=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)
    
    max_iters = config.max_epochs * len(train_loader)
    best_val_loss = float('inf')
    best_val_iou = 0.0
    start_epoch = 0
    
    # 训练循环
    for epoch in range(start_epoch, config.max_epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1}/{config.max_epochs}")
        logger.info(f"{'='*70}")
        
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, config.max_epochs, max_iters,
            config.base_lr, config.device, config,
            val_loader=val_loader, val_metrics=val_metrics,
            best_val_iou_list=[best_val_iou],
            output_dir=os.path.join(config.output_dir, 'checkpoints'),
            logger=logger,
            swanlab_run=swanlab_run
        )
        
        logger.info(f"训练损失: {train_loss:.4f}")
        
        # 记录epoch级别的训练损失到 SwanLab
        if swanlab_run is not None:
            log_dict = {
                'train/epoch_loss': train_loss,
                'train/epoch': epoch + 1
            }
            swanlab.log(log_dict)
        
        # Epoch级别的验证
        if (epoch + 1) % config.eval_interval == 0:
            logger.info("\n进行验证...")
            val_loss, val_score = validate(model, val_loader, val_metrics, config.device, swanlab_run=swanlab_run)
            
            current_iou = val_score['Mean IoU']
            logger.info(f"验证损失: {val_loss:.4f}")
            logger.info(f"验证 Mean IoU: {current_iou:.4f}")
            
            # 记录epoch级别的验证指标到 SwanLab
            if swanlab_run is not None:
                val_log_dict = {
                    'val/epoch_loss': val_loss,
                    'val/epoch_miou': current_iou,
                    'val/epoch_overall_acc': val_score['Overall Acc'],
                    'val/epoch_mean_acc': val_score['Mean Acc'],
                    'val/epoch_mean_prec': val_score['Mean Prec'],
                    'val/epoch': epoch + 1
                }
                swanlab.log(val_log_dict)
            
            # 保存最佳模型
            is_best_iou = current_iou > best_val_iou
            if is_best_iou:
                best_val_iou = current_iou
                checkpoint_path = os.path.join(
                    config.output_dir, 'checkpoints', 'best_model.pth'
                )
                try:
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, checkpoint_path,
                        is_best=True, best_val_iou=best_val_iou, logger=logger
                    )
                    logger.info(f"✓ 保存最佳模型 (Mean IoU: {best_val_iou:.4f})")
                except Exception as e:
                    logger.warning(f"保存最佳模型失败，但训练继续: {e}")
            
            # 保存最后一个模型
            if config.save_last:
                checkpoint_path = os.path.join(
                    config.output_dir, 'checkpoints', 'last_model.pth'
                )
                try:
                    save_checkpoint(
                        model, optimizer, epoch, val_loss, checkpoint_path,
                        is_best=False, best_val_iou=best_val_iou, logger=logger
                    )
                except Exception as e:
                    logger.warning(f"保存最后一个模型失败，但训练继续: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("训练完成！")
    logger.info(f"最佳验证 Mean IoU: {best_val_iou:.4f}")
    logger.info(f"模型保存在: {config.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
