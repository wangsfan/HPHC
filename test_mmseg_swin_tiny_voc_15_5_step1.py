"""
可视化 Step1 模型结构并测试 Step0 最佳模型在 Step1 训练集上的 mIoU

功能：
1. 可视化模型内部结构（层次结构、参数统计等）
2. 加载 Step0 最佳模型权重（16类）
3. 在 Step1 训练集上测试 mIoU（21类数据集）
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from PIL import Image

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
# 注意：处理可能的 charset_normalizer 循环导入问题
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 尝试修复 charset_normalizer 的导入问题（如果存在）
import sys
if 'charset_normalizer' in sys.modules:
    # 清除 charset_normalizer 相关的模块缓存
    modules_to_remove = [k for k in sys.modules.keys() if 'charset_normalizer' in k]
    for mod in modules_to_remove:
        del sys.modules[mod]

try:
    from mmcv import Config
    from mmseg.models import build_segmentor
except (AttributeError, ImportError) as e:
    if 'charset_normalizer' in str(e) or 'md__mypyc' in str(e):
        print("=" * 70)
        print("警告: 检测到 charset_normalizer 导入问题")
        print("=" * 70)
        print("这通常是由于 charset_normalizer 包版本不兼容导致的。")
        print("建议运行以下命令修复:")
        print("  pip install --upgrade --force-reinstall charset-normalizer")
        print("或者:")
        print("  pip install --upgrade --force-reinstall requests")
        print("=" * 70)
        raise ImportError(
            "charset_normalizer 导入失败。请运行: "
            "pip install --upgrade --force-reinstall charset-normalizer"
        ) from e
    else:
        raise

# 导入 WILSON 数据加载器
from data_loader import create_dataloaders, get_task_dict, get_task_labels
from load_voc_dataset import VOCDatasetConfig

# 导入评估指标
from metrics import StreamSegMetrics


# ==================== 模型可视化函数 ====================
def print_model_structure(model, logger=None):
    """打印模型的层次结构"""
    def print_module(module, prefix="", max_depth=3, current_depth=0):
        """递归打印模块结构"""
        if current_depth >= max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            num_params = sum(p.numel() for p in child.parameters())
            
            if num_params > 0:
                if logger:
                    logger.info(f"{'  ' * current_depth}{full_name}: {type(child).__name__} ({num_params:,} params)")
                else:
                    print(f"{'  ' * current_depth}{full_name}: {type(child).__name__} ({num_params:,} params)")
            
            # 递归打印子模块
            if len(list(child.children())) > 0 and current_depth < max_depth - 1:
                print_module(child, full_name, max_depth, current_depth + 1)
    
    if logger:
        logger.info("\n" + "=" * 70)
        logger.info("模型结构:")
        logger.info("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("模型结构:")
        print("=" * 70)
    
    print_module(model, max_depth=4)


def print_model_summary(model, logger=None):
    """打印模型参数统计摘要"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 按模块统计参数
    module_params = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += param.numel()
    
    if logger:
        logger.info("\n" + "=" * 70)
        logger.info("模型参数统计:")
        logger.info("=" * 70)
        logger.info(f"总参数数: {total_params:,}")
        logger.info(f"可训练参数数: {trainable_params:,}")
        logger.info(f"不可训练参数数: {total_params - trainable_params:,}")
        logger.info("\n各模块参数统计:")
        for module_name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {module_name}: {params:,} ({params/total_params*100:.2f}%)")
    else:
        print("\n" + "=" * 70)
        print("模型参数统计:")
        print("=" * 70)
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        print(f"不可训练参数数: {total_params - trainable_params:,}")
        print("\n各模块参数统计:")
        for module_name, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            print(f"  {module_name}: {params:,} ({params/total_params*100:.2f}%)")


def visualize_backbone_features(model, logger=None):
    """可视化 backbone 的结构"""
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if logger:
            logger.info("\n" + "=" * 70)
            logger.info("Backbone (Swin-Tiny) 结构:")
            logger.info("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("Backbone (Swin-Tiny) 结构:")
            print("=" * 70)
        
        # 打印 Swin Transformer 的关键配置
        if hasattr(backbone, 'embed_dim'):
            if logger:
                logger.info(f"  Embedding Dimension: {backbone.embed_dim}")
                logger.info(f"  Number of Layers: {backbone.num_layers}")
                if hasattr(backbone, 'layers'):
                    for i, layer in enumerate(backbone.layers):
                        logger.info(f"  Layer {i}: {len(layer.blocks)} blocks")
            else:
                print(f"  Embedding Dimension: {backbone.embed_dim}")
                print(f"  Number of Layers: {backbone.num_layers}")
                if hasattr(backbone, 'layers'):
                    for i, layer in enumerate(backbone.layers):
                        print(f"  Layer {i}: {len(layer.blocks)} blocks")


def visualize_decode_head(model, logger=None):
    """可视化 decode head 的结构"""
    if hasattr(model, 'decode_head'):
        decode_head = model.decode_head
        if logger:
            logger.info("\n" + "=" * 70)
            logger.info("Decode Head (UPerNet) 结构:")
            logger.info("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("Decode Head (UPerNet) 结构:")
            print("=" * 70)
        
        if hasattr(decode_head, 'in_channels'):
            if logger:
                logger.info(f"  Input Channels: {decode_head.in_channels}")
            else:
                print(f"  Input Channels: {decode_head.in_channels}")
        
        if hasattr(decode_head, 'channels'):
            if logger:
                logger.info(f"  Feature Channels: {decode_head.channels}")
            else:
                print(f"  Feature Channels: {decode_head.channels}")
        
        if hasattr(decode_head, 'num_classes'):
            if logger:
                logger.info(f"  Number of Classes: {decode_head.num_classes}")
            else:
                print(f"  Number of Classes: {decode_head.num_classes}")
        
        # 检查是否有 PSP 模块
        if hasattr(decode_head, 'psp_modules'):
            if logger:
                logger.info(f"  PSP Module: 存在 (pool_scales: {decode_head.psp_modules.pool_scales})")
            else:
                print(f"  PSP Module: 存在 (pool_scales: {decode_head.psp_modules.pool_scales})")
        
        # 检查是否有 FPN 模块
        if hasattr(decode_head, 'lateral_convs'):
            if logger:
                logger.info(f"  FPN Module: 存在 ({len(decode_head.lateral_convs)} lateral convs)")
            else:
                print(f"  FPN Module: 存在 ({len(decode_head.lateral_convs)} lateral convs)")


def visualize_auxiliary_head(model, logger=None):
    """可视化 auxiliary head 的结构"""
    if hasattr(model, 'auxiliary_head') and model.auxiliary_head is not None:
        aux_head = model.auxiliary_head
        if logger:
            logger.info("\n" + "=" * 70)
            logger.info("Auxiliary Head (FCN) 结构:")
            logger.info("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("Auxiliary Head (FCN) 结构:")
            print("=" * 70)
        
        if hasattr(aux_head, 'in_channels'):
            if logger:
                logger.info(f"  Input Channels: {aux_head.in_channels}")
            else:
                print(f"  Input Channels: {aux_head.in_channels}")
        
        if hasattr(aux_head, 'channels'):
            if logger:
                logger.info(f"  Feature Channels: {aux_head.channels}")
            else:
                print(f"  Feature Channels: {aux_head.channels}")
        
        if hasattr(aux_head, 'num_classes'):
            if logger:
                logger.info(f"  Number of Classes: {aux_head.num_classes}")
            else:
                print(f"  Number of Classes: {aux_head.num_classes}")


# ==================== 加载 Step0 权重函数 ====================
def load_step0_weights(model, step0_checkpoint_path, device, logger=None):
    """加载 Step0 模型权重到新模型（除了分类头）"""
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
    else:
        print(f"\n加载 Step0 最佳模型权重...")
        print(f"✓ Step0 检查点: {step0_checkpoint_path}")
    
    checkpoint = torch.load(step0_checkpoint_path, map_location=device)
    
    # 获取模型状态字典
    if 'model_state_dict' in checkpoint:
        step0_state_dict = checkpoint['model_state_dict']
        if logger:
            logger.info(f"  检查点信息:")
            if 'epoch' in checkpoint:
                logger.info(f"    Epoch: {checkpoint['epoch']}")
            if 'best_val_iou' in checkpoint:
                logger.info(f"    Best IoU: {checkpoint['best_val_iou']:.4f}")
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
        logger.info(f"  复制了 {len(copied_keys)} 个键")
        if skipped_keys:
            logger.info(f"  跳过 {len(skipped_keys)} 个不匹配的键（主要是分类头）")
    else:
        print(f"✓ Step0 权重加载成功（backbone 和大部分层）")
        print(f"  复制了 {len(copied_keys)} 个键")
        if skipped_keys:
            print(f"  跳过 {len(skipped_keys)} 个不匹配的键（主要是分类头）")
    
    return step0_state_dict


def expand_classification_head(model, step0_state_dict, old_num_classes=16, new_num_classes=21, logger=None):
    """扩展分类头从 old_num_classes 到 new_num_classes，并复制权重"""
    if logger:
        logger.info(f"\n扩展分类头: {old_num_classes}类 -> {new_num_classes}类")
    else:
        print(f"\n扩展分类头: {old_num_classes}类 -> {new_num_classes}类")
    
    # 扩展 decode_head 分类头
    if hasattr(model, 'decode_head') and model.decode_head is not None:
        decode_head = model.decode_head
        
        if hasattr(decode_head, 'conv_seg'):
            conv_seg = decode_head.conv_seg
            
            old_weight_key = 'decode_head.conv_seg.weight'
            if old_weight_key in step0_state_dict:
                old_weight = step0_state_dict[old_weight_key]
                
                channels = old_weight.shape[1]
                new_weight = torch.zeros(new_num_classes, channels, 1, 1, device=old_weight.device, dtype=old_weight.dtype)
                
                # 复制旧类别权重 (0-15)
                new_weight[:old_num_classes] = old_weight
                
                # 新类别使用 step0 所有类别的平均权重
                avg_weight = old_weight.mean(dim=0, keepdim=True)
                for i in range(old_num_classes, new_num_classes):
                    new_weight[i] = avg_weight[0]
                
                conv_seg.weight.data = new_weight
                
                if logger:
                    logger.info(f"✓ decode_head 分类头权重已复制")
                else:
                    print(f"✓ decode_head 分类头权重已复制")
        
        # 处理 bias
        if hasattr(decode_head, 'conv_seg') and decode_head.conv_seg.bias is not None:
            old_bias_key = 'decode_head.conv_seg.bias'
            if old_bias_key in step0_state_dict:
                old_bias = step0_state_dict[old_bias_key]
                new_bias = torch.zeros(new_num_classes, device=old_bias.device, dtype=old_bias.dtype)
                new_bias[:old_num_classes] = old_bias
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
                
                new_weight[:old_num_classes] = old_weight
                avg_weight = old_weight.mean(dim=0, keepdim=True)
                for i in range(old_num_classes, new_num_classes):
                    new_weight[i] = avg_weight[0]
                
                conv_seg.weight.data = new_weight
                
                if logger:
                    logger.info(f"✓ auxiliary_head 分类头权重已复制")
                else:
                    print(f"✓ auxiliary_head 分类头权重已复制")
        
        # 处理 bias
        if hasattr(auxiliary_head, 'conv_seg') and auxiliary_head.conv_seg.bias is not None:
            old_bias_key = 'auxiliary_head.conv_seg.bias'
            if old_bias_key in step0_state_dict:
                old_bias = step0_state_dict[old_bias_key]
                new_bias = torch.zeros(new_num_classes, device=old_bias.device, dtype=old_bias.dtype)
                new_bias[:old_num_classes] = old_bias
                avg_bias = old_bias.mean()
                new_bias[old_num_classes:] = avg_bias
                auxiliary_head.conv_seg.bias.data = new_bias


# ==================== 特征图捕获和可视化 ====================
class FeatureHook:
    """用于捕获模型中间特征的Hook类"""
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def register_hook(self, name, module):
        """注册hook到指定模块"""
        def hook_fn(module, input, output):
            self.features[name] = output.detach()
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_features(self):
        """清空特征"""
        self.features = {}


def visualize_feature_map(feature, method='pca', save_path=None):
    """
    可视化特征图
    
    Args:
        feature: 特征图 [C, H, W] 或 [B, C, H, W]
        method: 可视化方法 ('pca', 'mean', 'max', 'first3')
        save_path: 保存路径
    
    Returns:
        vis_image: 可视化图像 [H, W, 3] (numpy array, 0-255)
    """
    # 如果是batch，取第一个
    if feature.dim() == 4:
        feature = feature[0]
    
    # 确保是 [C, H, W]
    assert feature.dim() == 3, f"Expected 3D tensor [C, H, W], got {feature.shape}"
    
    C, H, W = feature.shape
    feature_np = feature.cpu().numpy()
    
    if method == 'pca':
        # 使用PCA降维到3通道
        try:
            from sklearn.decomposition import PCA
            feature_flat = feature_np.reshape(C, -1).T  # [H*W, C]
            pca = PCA(n_components=3)
            feature_pca = pca.fit_transform(feature_flat)  # [H*W, 3]
            vis_image = feature_pca.reshape(H, W, 3)
            # 归一化到 [0, 1]
            vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
        except ImportError:
            # 如果sklearn不可用，使用前3个主成分的简单近似
            feature_flat = feature_np.reshape(C, -1)  # [C, H*W]
            # 计算前3个通道的方差，选择方差最大的3个通道
            variances = np.var(feature_flat, axis=1)
            top3_indices = np.argsort(variances)[-3:][::-1]
            vis_image = feature_np[top3_indices].transpose(1, 2, 0)  # [H, W, 3]
            vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
    
    elif method == 'mean':
        # 取通道平均
        vis_image = feature_np.mean(axis=0)  # [H, W]
        vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
        vis_image = np.stack([vis_image] * 3, axis=-1)  # [H, W, 3]
    
    elif method == 'max':
        # 取通道最大值
        vis_image = feature_np.max(axis=0)  # [H, W]
        vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
        vis_image = np.stack([vis_image] * 3, axis=-1)  # [H, W, 3]
    
    elif method == 'first3':
        # 取前3个通道
        vis_image = feature_np[:3].transpose(1, 2, 0)  # [H, W, 3]
        vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8)
    
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    # 转换为 [0, 255] 的uint8
    vis_image = (vis_image * 255).astype(np.uint8)
    
    # 保存图片
    if save_path:
        Image.fromarray(vis_image).save(save_path)
    
    return vis_image


def analyze_backbone_features(backbone_features, input_size=(512, 512), logger=None):
    """
    分析backbone特征的实际维度
    
    Args:
        backbone_features: tuple of 4 features from backbone
        input_size: 输入图像尺寸 (H, W)
        logger: 日志记录器
    
    Returns:
        analysis: 分析结果字典
    """
    analysis = {}
    
    for i, feat in enumerate(backbone_features):
        B, C, H, W = feat.shape
        resolution_ratio = (H / input_size[0], W / input_size[1])
        analysis[f'stage{i}'] = {
            'shape': feat.shape,
            'channels': C,
            'spatial_size': (H, W),
            'resolution_ratio': resolution_ratio,
            'total_elements': feat.numel()
        }
    
    if logger:
        logger.info("\n" + "=" * 70)
        logger.info("Swin-Tiny Backbone 特征维度分析:")
        logger.info("=" * 70)
        logger.info(f"输入图像尺寸: {input_size}")
        for stage_name, info in analysis.items():
            logger.info(f"\n{stage_name.upper()}:")
            logger.info(f"  形状: {info['shape']}")
            logger.info(f"  通道数: {info['channels']}")
            logger.info(f"  空间尺寸: {info['spatial_size']}")
            logger.info(f"  分辨率比例: {info['resolution_ratio']}")
            logger.info(f"  总元素数: {info['total_elements']:,}")
        
        # 分析是否需要降维
        logger.info("\n" + "-" * 70)
        logger.info("降维分析:")
        logger.info("-" * 70)
        channels = [info['channels'] for info in analysis.values()]
        max_channels = max(channels)
        min_channels = min(channels)
        logger.info(f"  通道数范围: {min_channels} - {max_channels}")
        logger.info(f"  通道数差异: {max_channels / min_channels:.2f}x")
        
        if max_channels / min_channels > 2:
            logger.info("  建议: 需要降维，通道数差异较大")
            logger.info("  方案: 统一到256通道（参考SIPE V2）")
            logger.info("  优势: 保证各层权重对等，避免深层特征占主导")
        else:
            logger.info("  建议: 可以不降维，直接使用原始特征")
    
    return analysis


def create_side_branches(backbone_features, device='cuda', unified_channels=256, logger=None):
    """
    创建Side Branches用于多尺度特征融合
    
    根据实际backbone特征维度动态创建，参考SIPE V2的做法统一通道数
    
    Args:
        backbone_features: tuple of 4 features from backbone
        device: 设备
        unified_channels: 统一后的通道数（默认256）
        logger: 日志记录器
    
    Returns:
        side_branches: dict of side branch conv layers
    """
    x0, x1, x2, x3 = backbone_features
    
    # 获取实际通道数
    c0, c1, c2, c3 = x0.shape[1], x1.shape[1], x2.shape[1], x3.shape[1]
    
    if logger:
        logger.info(f"\n创建Side Branches:")
        logger.info(f"  Stage 0: {c0} channels -> {unified_channels} channels")
        logger.info(f"  Stage 1: {c1} channels -> {unified_channels} channels")
        logger.info(f"  Stage 2: {c2} channels -> {unified_channels} channels")
        logger.info(f"  Stage 3: {c3} channels -> {unified_channels} channels")
        logger.info(f"  统一通道数: {unified_channels} (参考SIPE V2)")
    
    # 创建1x1卷积层进行降维/升维
    side1 = nn.Conv2d(c0, unified_channels, 1, bias=False).to(device)   # Stage 0
    side2 = nn.Conv2d(c1, unified_channels, 1, bias=False).to(device)  # Stage 1
    side3 = nn.Conv2d(c2, unified_channels, 1, bias=False).to(device)  # Stage 2
    side4 = nn.Conv2d(c3, unified_channels, 1, bias=False).to(device)  # Stage 3
    
    # 初始化权重（使用Xavier初始化）
    for side in [side1, side2, side3, side4]:
        nn.init.xavier_uniform_(side.weight)
    
    return {'side1': side1, 'side2': side2, 'side3': side3, 'side4': side4}


def fuse_hierarchical_features(backbone_features, side_branches, device='cuda', logger=None):
    """
    融合多尺度特征（参考SIPE的做法）
    
    Args:
        backbone_features: tuple of 4 features from backbone
        side_branches: dict of side branch conv layers
        device: 设备
        logger: 日志记录器
    
    Returns:
        hie_fea: 融合后的特征 [B, 1024, H/16, W/16]
        side_features: dict of side features before fusion
    """
    x0, x1, x2, x3 = backbone_features  # Stage 0, 1, 2, 3
    
    # 从4个stage提取特征并统一通道数
    side1 = side_branches['side1'](x0.detach())  # [B, 256, H/4, W/4]
    side2 = side_branches['side2'](x1.detach())  # [B, 256, H/8, W/8]
    side3 = side_branches['side3'](x2.detach())  # [B, 256, H/16, W/16]
    side4 = side_branches['side4'](x3.detach())  # [B, 256, H/32, W/32]
    
    # 目标分辨率：Stage 2的分辨率（1/16，对于512x512输入是32x32）
    target_size = side3.shape[2:]  # (H/16, W/16)
    
    # 参考SIPE V2：在拼接前进行L2归一化
    side1_norm = F.normalize(side1, dim=1, p=2)
    side2_norm = F.normalize(side2, dim=1, p=2)
    side3_norm = F.normalize(side3, dim=1, p=2)
    side4_norm = F.normalize(side4, dim=1, p=2)
    
    # 将所有side features插值到Stage 2的分辨率并拼接
    hie_fea = torch.cat([
        F.interpolate(side1_norm, size=target_size, mode='bilinear', align_corners=False),
        F.interpolate(side2_norm, size=target_size, mode='bilinear', align_corners=False),
        F.interpolate(side3_norm, size=target_size, mode='bilinear', align_corners=False),
        F.interpolate(side4_norm, size=target_size, mode='bilinear', align_corners=False)
    ], dim=1)  # [B, 256*4=1024, H/16, W/16]
    
    if logger:
        logger.info(f"  多尺度特征融合:")
        logger.info(f"    Stage 0: {x0.shape} -> side1: {side1.shape} -> 插值到: {target_size}")
        logger.info(f"    Stage 1: {x1.shape} -> side2: {side2.shape} -> 插值到: {target_size}")
        logger.info(f"    Stage 2: {x2.shape} -> side3: {side3.shape} (目标分辨率)")
        logger.info(f"    Stage 3: {x3.shape} -> side4: {side4.shape} -> 插值到: {target_size}")
        logger.info(f"    融合后 hie_fea: {hie_fea.shape}")
    
    side_features = {
        'side1': side1,
        'side2': side2,
        'side3': side3,
        'side4': side4
    }
    
    return hie_fea, side_features


def create_cam_classifier(backbone_features, num_classes=21, device='cuda', logger=None):
    """
    创建CAM分类器（从Stage 3生成CAM）
    
    Args:
        backbone_features: tuple of 4 features from backbone
        num_classes: 类别数（包括背景）
        device: 设备
        logger: 日志记录器
    
    Returns:
        classifier: CAM分类器
    """
    x0, x1, x2, x3 = backbone_features
    stage3_channels = x3.shape[1]  # Stage 3的通道数（Swin-Tiny是768）
    
    # CAM分类器：从Stage 3生成CAM（20类，不含背景）
    # 参考SIPE的做法，CAM只生成前景类，背景类后续单独添加
    classifier = nn.Conv2d(stage3_channels, num_classes - 1, 1, bias=False).to(device)
    
    # 初始化权重（使用Xavier初始化）
    nn.init.xavier_uniform_(classifier.weight)
    
    if logger:
        logger.info(f"\n创建CAM分类器:")
        logger.info(f"  输入通道数: {stage3_channels} (Stage 3)")
        logger.info(f"  输出类别数: {num_classes - 1} (不含背景)")
        logger.info(f"  最终CAM类别数: {num_classes} (包含背景)")
    
    return classifier


def generate_cam(stage3_feature, classifier, input_size, logger=None):
    """
    从Stage 3生成初始CAM并归一化
    
    Args:
        stage3_feature: Stage 3特征 [B, C, H, W]
        classifier: CAM分类器
        input_size: 输入图像尺寸 (H, W)
        logger: 日志记录器
    
    Returns:
        norm_cam: 归一化的CAM [B, num_classes, H, W]
        cam: 原始CAM [B, num_classes-1, H, W]
        score: 图像级分类分数 [B, num_classes-1, 1, 1]
    """
    # 生成CAM（20类，不含背景）
    cam = classifier(stage3_feature)  # [B, 20, H/32, W/32]
    
    # 图像级分类分数
    score = F.adaptive_avg_pool2d(cam, 1)  # [B, 20, 1, 1]
    
    # 归一化CAM
    norm_cam = F.relu(cam)
    norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
    
    # 添加背景类
    cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
    norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)  # [B, 21, H/32, W/32]
    
    # 插值到输入图像尺寸
    norm_cam = F.interpolate(
        norm_cam, 
        size=input_size, 
        mode='bilinear', 
        align_corners=False
    )  # [B, 21, H, W]
    
    if logger:
        logger.info(f"  CAM生成:")
        logger.info(f"    原始CAM尺寸: {cam.shape}")
        logger.info(f"    归一化后CAM尺寸: {norm_cam.shape}")
        logger.info(f"    图像级分数: {score.shape}")
    
    return norm_cam, cam, score


def visualize_cam_activation(cam, input_image, save_dir, class_names=None, logger=None):
    """
    可视化CAM激活图（每个类别的激活图）
    
    Args:
        cam: CAM激活图 [C, H, W] 或 [B, C, H, W]
        input_image: 输入图像 [H, W, 3] (numpy array, 0-255)
        save_dir: 保存目录
        class_names: 类别名称列表
        logger: 日志记录器
    """
    # 如果是batch，取第一个
    if cam.dim() == 4:
        cam = cam[0]
    
    # 确保是 [C, H, W]
    assert cam.dim() == 3, f"Expected 3D tensor [C, H, W], got {cam.shape}"
    
    C, H, W = cam.shape
    cam_np = cam.cpu().numpy()
    
    # 确保输入图像尺寸匹配
    if input_image.shape[:2] != (H, W):
        input_image = np.array(Image.fromarray(input_image).resize((W, H)))
    
    # 创建CAM可视化目录
    cam_dir = os.path.join(save_dir, 'cam_activations')
    os.makedirs(cam_dir, exist_ok=True)
    
    # 可视化每个类别的CAM激活图
    for class_id in range(C):
        class_cam = cam_np[class_id]  # [H, W]
        
        # 归一化到 [0, 1]
        class_cam_norm = (class_cam - class_cam.min()) / (class_cam.max() - class_cam.min() + 1e-8)
        
        # 使用热力图可视化（matplotlib已在文件开头导入）
        
        # 创建热力图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 左侧：原始图像
        axes[0].imshow(input_image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # 右侧：CAM激活图叠加在图像上
        axes[1].imshow(input_image)
        heatmap = axes[1].imshow(class_cam_norm, cmap='jet', alpha=0.5, interpolation='bilinear')
        axes[1].set_title(f'CAM - Class {class_id}' + (f' ({class_names[class_id]})' if class_names and class_id < len(class_names) else ''))
        axes[1].axis('off')
        plt.colorbar(heatmap, ax=axes[1])
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(cam_dir, f'cam_class_{class_id:02d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 也保存纯热力图（不叠加图像）
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(class_cam_norm, cmap='jet', interpolation='bilinear')
        ax.set_title(f'CAM Heatmap - Class {class_id}' + (f' ({class_names[class_id]})' if class_names and class_id < len(class_names) else ''))
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        save_path_heatmap = os.path.join(cam_dir, f'cam_class_{class_id:02d}_heatmap.png')
        plt.savefig(save_path_heatmap, dpi=150, bbox_inches='tight')
        plt.close()
    
    # 保存所有类别的综合可视化（前几个激活度最高的类别）
    # 找到每个像素激活度最高的类别
    max_cam = cam_np.max(axis=0)  # [H, W] - 每个像素的最大激活值
    argmax_cam = cam_np.argmax(axis=0)  # [H, W] - 每个像素激活度最高的类别
    
    # 创建类别可视化（使用伪彩色）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 最大激活类别
    axes[1].imshow(argmax_cam, cmap='tab20', vmin=0, vmax=C-1)
    axes[1].set_title('Max Activation Class')
    axes[1].axis('off')
    
    # 最大激活值
    im = axes[2].imshow(max_cam, cmap='hot', interpolation='bilinear')
    axes[2].set_title('Max Activation Value')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    save_path = os.path.join(cam_dir, 'cam_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"  ✓ CAM激活图已保存到 {cam_dir}")
        logger.info(f"    每个类别2张图: cam_class_XX.png (叠加图像) 和 cam_class_XX_heatmap.png (纯热力图)")
        logger.info(f"    综合可视化: cam_summary.png")


def test_with_feature_visualization(model, test_loader, metrics, device, output_dir, 
                                    max_samples=50, logger=None):
    """
    测试函数，同时可视化特征图
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        metrics: 评估指标
        device: 设备
        output_dir: 输出目录
        max_samples: 最大可视化样本数
        logger: 日志记录器
    """
    model.eval()
    metrics.reset()
    
    # 创建特征图保存目录
    feature_dir = os.path.join(output_dir, 'feature_maps')
    os.makedirs(feature_dir, exist_ok=True)
    
    # 注册hook捕获auxiliary head的特征（在分类层之前）
    hook = FeatureHook()
    aux_feat_captured = False
    
    if hasattr(model, 'auxiliary_head') and model.auxiliary_head is not None:
        aux_head = model.auxiliary_head
        # 找到auxiliary head的最后一个卷积层（分类层之前）
        if hasattr(aux_head, 'convs') and len(aux_head.convs) > 0:
            hook.register_hook('auxiliary_head_feat', aux_head.convs[-1])
            aux_feat_captured = True
        elif hasattr(aux_head, 'conv_seg'):
            # 在conv_seg之前注册hook，需要找到conv_seg的输入
            # 通常auxiliary head的输入是backbone的stage 2特征
            pass
    
    # 创建Side Branches和CAM分类器（先分析第一个batch的特征维度）
    side_branches = None
    cam_classifier = None
    sample_count = 0
    
    # VOC类别名称（用于CAM可视化）
    voc_class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing with Feature Visualization', leave=False)
        for batch_idx, (images, labels, l1h) in enumerate(pbar):
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
            
            # 直接调用backbone获取4层特征
            backbone_features = model.backbone(images)  # tuple of 4 features
            
            # 在第一个batch时分析特征维度并创建Side Branches
            if batch_idx == 0:
                if logger:
                    logger.info("\n分析Backbone特征维度...")
                
                # 分析特征维度
                feature_analysis = analyze_backbone_features(
                    backbone_features, 
                    input_size=images.shape[2:], 
                    logger=logger
                )
                
                # 创建Side Branches（基于实际特征维度）
                if logger:
                    logger.info("\n创建Side Branches用于多尺度特征融合...")
                side_branches = create_side_branches(backbone_features, device, logger=logger)
                if logger:
                    logger.info("✓ Side Branches创建成功")
                    logger.info("  统一通道数: 256")
                    logger.info("  目标分辨率: Stage 2 (1/16)")
                
                # 创建CAM分类器（基于实际特征维度）
                if logger:
                    logger.info("\n创建CAM分类器...")
                cam_classifier = create_cam_classifier(
                    backbone_features, 
                    num_classes=21,  # Step1有21类
                    device=device, 
                    logger=logger
                )
                if logger:
                    logger.info("✓ CAM分类器创建成功")
            
            # ==================== 多尺度特征融合（参考SIPE） ====================
            if side_branches is not None:
                hie_fea, side_features = fuse_hierarchical_features(
                    backbone_features, side_branches, device, logger if batch_idx == 0 else None
                )
            else:
                # 如果side_branches还没创建，跳过融合（不应该发生）
                hie_fea = None
                side_features = {}
            
            # 然后通过decode head进行推理
            seg_logits = model.encode_decode(images, img_metas)
            
            # 获取auxiliary head的特征（如果有）
            aux_feat = None
            if aux_feat_captured and 'auxiliary_head_feat' in hook.features:
                aux_feat = hook.features['auxiliary_head_feat']
            
            # 如果没有通过hook捕获，尝试从backbone的stage 2获取（auxiliary head通常使用stage 2）
            if aux_feat is None and len(backbone_features) > 2:
                aux_feat = backbone_features[2]  # Stage 2特征作为auxiliary head的输入
            
            # 获取预测结果（argmax）
            _, predictions = seg_logits.max(dim=1)  # [B, H, W]
            
            # 转换为numpy并更新评估指标
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            metrics.update(labels_np, predictions_np)
            
            # 可视化前max_samples个样本的特征图
            if sample_count < max_samples:
                for i in range(images.size(0)):
                    if sample_count >= max_samples:
                        break
                    
                    sample_id = sample_count
                    sample_dir = os.path.join(feature_dir, f'sample_{sample_id:03d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # 可视化backbone的4层特征
                    for stage_idx in range(len(backbone_features)):
                        feat = backbone_features[stage_idx]
                        # 如果是batch，取第一个
                        if feat.dim() == 4:
                            feat = feat[i]
                        
                        # 调整到统一尺寸以便可视化（128x128）
                        feat_resized = F.interpolate(
                            feat.unsqueeze(0), 
                            size=(128, 128), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                        
                        save_path = os.path.join(sample_dir, f'backbone_stage{stage_idx}.png')
                        visualize_feature_map(feat_resized, method='pca', save_path=save_path)
                    
                    # 可视化side features（降维后的特征）
                    if side_features:
                        for side_name in ['side1', 'side2', 'side3', 'side4']:
                            if side_name in side_features:
                                side_feat = side_features[side_name]
                                if side_feat.dim() == 4:
                                    side_feat = side_feat[i]
                                
                                # 调整到统一尺寸以便可视化（128x128）
                                side_feat_resized = F.interpolate(
                                    side_feat.unsqueeze(0),
                                    size=(128, 128),
                                    mode='bilinear',
                                    align_corners=False
                                ).squeeze(0)
                                
                                save_path = os.path.join(sample_dir, f'{side_name}_feat.png')
                                visualize_feature_map(side_feat_resized, method='pca', save_path=save_path)
                    
                    # ==================== 生成并可视化CAM ====================
                    if cam_classifier is not None:
                        # 获取Stage 3特征（用于生成CAM）
                        stage3_feat = backbone_features[3]  # Stage 3 (x3)
                        if stage3_feat.dim() == 4:
                            stage3_feat_single = stage3_feat[i:i+1]  # 保持batch维度
                        else:
                            stage3_feat_single = stage3_feat.unsqueeze(0)
                        
                        # 生成CAM
                        input_size = images.shape[2:]  # (H, W)
                        norm_cam, cam, score = generate_cam(
                            stage3_feat_single, 
                            cam_classifier, 
                            input_size, 
                            logger if sample_count == 0 else None
                        )
                        
                        # 准备输入图像用于可视化（反归一化）
                        input_image_np = images[i].cpu().numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        input_image_np = std * input_image_np + mean
                        input_image_np = np.clip(input_image_np * 255, 0, 255).astype(np.uint8)
                        
                        # 可视化CAM激活图
                        visualize_cam_activation(
                            norm_cam, 
                            input_image_np, 
                            sample_dir, 
                            class_names=voc_class_names,
                            logger=logger if sample_count == 0 else None
                        )
                    
                    # 可视化hie_fea（融合后的多尺度特征）
                    if hie_fea is not None:
                        hie_fea_single = hie_fea[i] if hie_fea.dim() == 4 else hie_fea
                        
                        # hie_fea已经是Stage 2的分辨率（32x32），调整到128x128以便可视化
                        hie_fea_resized = F.interpolate(
                            hie_fea_single.unsqueeze(0),
                            size=(128, 128),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        
                        save_path = os.path.join(sample_dir, 'hie_fea_fused.png')
                        visualize_feature_map(hie_fea_resized, method='pca', save_path=save_path)
                    
                    # 可视化auxiliary head特征
                    if aux_feat is not None:
                        aux_feat_single = aux_feat[i] if aux_feat.dim() == 4 else aux_feat
                        
                        aux_feat_resized = F.interpolate(
                            aux_feat_single.unsqueeze(0),
                            size=(128, 128),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        
                        save_path = os.path.join(sample_dir, 'auxiliary_head_feat.png')
                        visualize_feature_map(aux_feat_resized, method='pca', save_path=save_path)
                    
                    # 也保存原始图像
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    # 反归一化（假设是ImageNet归一化）
                    img_np = np.clip(img_np, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(os.path.join(sample_dir, 'input_image.png'))
                    
                    sample_count += 1
                    
                    # 清空特征以便下次使用
                    hook.clear_features()
    
    # 移除hooks
    hook.remove_hooks()
    
    # 计算评估得分
    score = metrics.get_results()
    
    if logger:
        logger.info(f"\n✓ 特征图可视化完成，已保存 {sample_count} 个样本到 {feature_dir}")
        logger.info(f"  每个样本包含:")
        logger.info(f"    - 4个backbone原始特征图 (backbone_stage0-3.png)")
        logger.info(f"    - 4个side features降维特征图 (side1-4_feat.png)")
        logger.info(f"    - 1个融合后的多尺度特征图 (hie_fea_fused.png)")
        logger.info(f"    - 1个auxiliary head特征图 (auxiliary_head_feat.png)")
        logger.info(f"    - 1个输入图像 (input_image.png)")
    
    return score


# ==================== 测试函数 ====================
def test(model, test_loader, metrics, device, logger=None):
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


# ==================== 主函数 ====================
def main():
    """主测试函数"""
    print("=" * 70)
    print("Step1 模型可视化与测试 - Step0 最佳模型在 Step1 训练集上的 mIoU")
    print("=" * 70)
    
    # 配置
    step0_checkpoint = os.path.join(
        current_dir,
        'outputs/mmseg_swin_tiny_voc_15-5_step0/checkpoint_iter_36500_0.7605_best.pth'
    )
    
    task = '15-5'
    step = 1
    data_root = os.path.join(current_dir, 'WILSON', 'data')
    crop_size = 512
    batch_size = 1  # 测试时使用 batch_size=1
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 设置日志
    log_file = os.path.join(current_dir, 'outputs', 'test_step1_visualization.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
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
    logger.info(f"  Step0 检查点: {step0_checkpoint}")
    logger.info(f"  任务: {task} Step {step}")
    logger.info(f"  设备: {device}")
    logger.info("=" * 70)
    
    # ==================== 加载数据集 ====================
    logger.info("\n加载数据集...")
    
    dataset_config = VOCDatasetConfig()
    dataset_config.task = task
    dataset_config.step = step
    dataset_config.batch_size = batch_size
    dataset_config.num_workers = num_workers
    dataset_config.crop_size = crop_size
    dataset_config.crop_size_val = crop_size
    dataset_config.data_root = data_root
    
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
    except Exception as e:
        logger.error(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 加载模型 ====================
    logger.info("\n加载模型...")
    
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
    except Exception as e:
        logger.error(f"✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 移动到指定设备
    if device == 'cuda':
        logger.info("移动模型到 GPU...")
        model = model.cuda()
        logger.info("✓ 模型已移动到 GPU")
    else:
        logger.info("使用 CPU")
        model = model.cpu()
    
    # ==================== 模型可视化 ====================
    logger.info("\n" + "=" * 70)
    logger.info("模型可视化")
    logger.info("=" * 70)
    
    # 打印模型参数统计
    print_model_summary(model, logger)
    
    # 打印模型结构
    print_model_structure(model, logger)
    
    # 可视化各个组件
    visualize_backbone_features(model, logger)
    visualize_decode_head(model, logger)
    visualize_auxiliary_head(model, logger)
    
    # ==================== 加载 Step0 权重并扩展分类头 ====================
    try:
        # 加载 Step0 权重（除了分类头）
        step0_state_dict = load_step0_weights(
            model, step0_checkpoint, device, logger
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
    
    # ==================== 创建评估指标对象 ====================
    logger.info("\n创建评估指标对象...")
    train_metrics = StreamSegMetrics(n_classes=n_classes)
    logger.info(f"✓ 评估指标对象创建成功 (类别数: {n_classes})")
    
    # ==================== 在训练集上测试（带特征图可视化） ====================
    logger.info("\n" + "=" * 70)
    logger.info("在 Step1 训练集上进行评估（带特征图可视化）...")
    logger.info("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(current_dir, 'outputs', 'test_step1_visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    train_score = test_with_feature_visualization(
        model, train_loader, train_metrics, device, output_dir, 
        max_samples=50, logger=logger
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Step1 训练集评估结果:")
    logger.info("=" * 70)
    logger.info(f"  Overall Accuracy: {train_score['Overall Acc']:.4f}")
    logger.info(f"  Mean Accuracy: {train_score['Mean Acc']:.4f}")
    logger.info(f"  Mean Precision: {train_score['Mean Prec']:.4f}")
    logger.info(f"  Mean IoU: {train_score['Mean IoU']:.4f}")
    
    logger.info("\n各类别IoU:")
    for class_id, iou in train_score['Class IoU'].items():
        if iou != "X":
            logger.info(f"  类别 {class_id}: {iou:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()


