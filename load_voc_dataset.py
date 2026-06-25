"""
VOC 数据集加载脚本
用于在 WILSS 文件夹路径下加载 VOC 实验数据集

数据集路径结构：
WILSS/
  WILSON/
    data/
      voc/
        JPEGImages/          # 原始图像
        SegmentationClassAug/ # 分割标签
        splits/               # 分割文件
          train_aug.txt
          val.txt
        voc_1h_labels_train.npy
        voc_1h_labels_val.npy
        {task}/              # 任务目录（如 15-5, 19-1等）
          train-{step}.npy   # 训练集索引
          val-{step}.npy     # 验证集索引
          test_on_val-{step}.npy
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加 WILSON 目录到路径，以便导入 data_loader
current_dir = os.path.dirname(os.path.abspath(__file__))
wilson_dir = os.path.join(current_dir, 'WILSON')
sys.path.insert(0, wilson_dir)

from data_loader import get_dataset, create_dataloaders, get_task_dict, get_task_labels


class VOCDatasetConfig:
    """VOC数据集配置类"""
    def __init__(self):
        # 数据集基本信息
        self.dataset = 'voc'           # 数据集名称
        self.task = '15-5'             # 任务名称: '19-1', '15-5', '15-1', '10-5'
        self.step = 0                   # 当前训练步骤
        
        # 数据根目录 - 相对于 WILSS 文件夹
        # 默认路径: WILSS/WILSON/data
        self.data_root = os.path.join(current_dir, 'WILSON', 'data')
        
        # 图像尺寸配置
        self.crop_size = 512            # 训练时裁剪大小
        self.crop_size_val = 512        # 验证时裁剪大小
        
        # 数据加载配置
        self.batch_size = 8             # batch size
        self.num_workers = 4            # 数据加载线程数
        
        # 其他选项
        self.overlap = False            # 是否使用重叠
        self.no_mask = False            # 是否不使用mask
        self.weakly = False             # 是否使用弱监督
        self.val_on_trainset = False    # 是否在训练集上验证
        self.pseudo = None              # 伪标签路径前缀（可选）


def print_dataset_info(opts):
    """打印数据集信息"""
    print("=" * 70)
    print("VOC 数据集配置信息")
    print("=" * 70)
    print(f"数据集名称: {opts.dataset.upper()}")
    print(f"任务类型: {opts.task}")
    print(f"当前步骤: {opts.step}")
    print(f"数据根目录: {opts.data_root}")
    print(f"图像尺寸: {opts.crop_size} x {opts.crop_size_val}")
    print(f"Batch Size: {opts.batch_size}")
    print(f"Num Workers: {opts.num_workers}")
    print(f"重叠模式: {opts.overlap}")
    print(f"使用Mask: {not opts.no_mask}")
    print(f"弱监督: {opts.weakly}")
    print("=" * 70)
    
    # 获取任务配置
    try:
        step_dict = get_task_dict(opts.dataset, opts.task, opts.step)
        labels, labels_old, path_base = get_task_labels(opts.dataset, opts.task, opts.step)
        
        print("\n类别信息:")
        print(f"  旧类别数: {len(labels_old)}")
        print(f"  新类别数: {len(labels)}")
        print(f"  总类别数: {len(labels_old) + len(labels)}")
        
        print("\n步骤类别分布:")
        for step, classes in sorted(step_dict.items()):
            print(f"  Step {step}: {len(classes)} 个类别")
            if len(classes) <= 10:
                print(f"    类别ID: {classes}")
        
        print("\n当前步骤类别:")
        current_classes = step_dict[opts.step]
        print(f"  类别ID: {current_classes}")
        
    except Exception as e:
        print(f"获取任务配置时出错: {e}")


def check_dataset_paths(opts):
    """检查数据集路径是否存在"""
    print("\n检查数据集路径...")
    
    voc_root = os.path.join(opts.data_root, 'voc')
    required_paths = {
        'VOC根目录': voc_root,
        '图像目录': os.path.join(voc_root, 'JPEGImages'),
        '标签目录': os.path.join(voc_root, 'SegmentationClassAug'),
        '分割文件目录': os.path.join(voc_root, 'splits'),
        '训练分割文件': os.path.join(voc_root, 'splits', 'train_aug.txt'),
        '验证分割文件': os.path.join(voc_root, 'splits', 'val.txt'),
        '训练标签文件': os.path.join(voc_root, 'voc_1h_labels_train.npy'),
        '验证标签文件': os.path.join(voc_root, 'voc_1h_labels_val.npy'),
    }
    
    # 任务特定路径
    path_base = os.path.join(opts.data_root, f'voc/{opts.task}')
    if opts.overlap:
        path_base += '-ov'
    
    required_paths[f'任务目录'] = path_base
    required_paths[f'训练索引文件'] = os.path.join(path_base, f'train-{opts.step}.npy')
    required_paths[f'验证索引文件'] = os.path.join(path_base, f'val-{opts.step}.npy')
    required_paths[f'测试索引文件'] = os.path.join(path_base, f'test_on_val-{opts.step}.npy')
    
    all_exist = True
    for name, path in required_paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist


def load_and_display_dataset(opts):
    """加载并显示数据集信息"""
    print("\n" + "=" * 70)
    print("加载数据集...")
    print("=" * 70)
    
    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader, n_classes = create_dataloaders(
            opts,
            distributed=False
        )
        
        print(f"\n✓ 数据集加载成功!")
        print(f"\n数据集大小:")
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  测试集: {len(test_loader.dataset)} 样本")
        print(f"  总类别数: {n_classes}")
        
        # 获取一个batch的数据示例
        print("\n获取数据示例...")
        for images, labels, l1h in train_loader:
            print(f"\nBatch信息:")
            print(f"  Images shape: {images.shape}")      # (B, 3, H, W)
            print(f"  Labels shape: {labels.shape}")      # (B, H, W)
            print(f"  One-hot labels shape: {l1h.shape}") # (B, C)
            print(f"  Images dtype: {images.dtype}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Labels unique values: {torch.unique(labels)}")
            break  # 只显示第一个batch
        
        return train_loader, val_loader, test_loader
        
    except FileNotFoundError as e:
        print(f"\n✗ 数据集加载失败: {e}")
        print("\n请检查:")
        print("  1. 数据集路径是否正确")
        print("  2. 索引文件是否存在")
        print("  3. 数据集文件是否完整")
        return None, None, None
    except Exception as e:
        print(f"\n✗ 数据集加载时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def visualize_sample(dataset, index=0, save_path=None):
    """可视化数据集样本"""
    try:
        img, lbl, l1h = dataset[index]
        
        # 如果是tensor，需要反归一化和转换
        if isinstance(img, torch.Tensor):
            # 反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
        else:
            img = np.array(img)
        
        # 转换标签
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.numpy()
        else:
            lbl = np.array(lbl)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原始图像
        axes[0].imshow(img)
        axes[0].set_title(f'原始图像 (样本 {index})')
        axes[0].axis('off')
        
        # 显示标签（使用颜色映射）
        axes[1].imshow(lbl, cmap='tab20', vmin=0, vmax=20)
        axes[1].set_title(f'分割标签 (样本 {index})')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # 打印信息
        unique_labels = np.unique(lbl)
        print(f"\n样本 {index} 信息:")
        print(f"  图像尺寸: {img.shape}")
        print(f"  标签尺寸: {lbl.shape}")
        print(f"  标签中的类别: {unique_labels}")
        if isinstance(l1h, torch.Tensor):
            active_classes = torch.nonzero(l1h).squeeze()
            if active_classes.dim() == 0:
                active_classes = [active_classes.item()]
            else:
                active_classes = active_classes.tolist()
            print(f"  图像级标签 (one-hot): {len(active_classes)} 个类别激活")
        
    except Exception as e:
        print(f"可视化样本时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("=" * 70)
    print("VOC 数据集加载脚本")
    print("=" * 70)
    
    # 创建配置
    opts = VOCDatasetConfig()
    
    # 可以在这里修改配置
    # opts.task = '19-1'  # 修改任务
    # opts.step = 0       # 修改步骤
    # opts.data_root = '/path/to/your/data'  # 修改数据路径
    
    # 打印配置信息
    print_dataset_info(opts)
    
    # 检查路径
    paths_ok = check_dataset_paths(opts)
    
    if not paths_ok:
        print("\n⚠ 警告: 部分路径不存在，可能无法正常加载数据集")
        response = input("\n是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 加载数据集
    train_loader, val_loader, test_loader = load_and_display_dataset(opts)
    
    if train_loader is None:
        print("\n数据集加载失败，退出")
        return
    
    # 可视化示例（可选）
    print("\n" + "=" * 70)
    response = input("是否可视化一个样本? (y/n): ")
    if response.lower() == 'y':
        try:
            # 可视化训练集的一个样本
            visualize_sample(train_loader.dataset, index=0)
        except Exception as e:
            print(f"可视化失败: {e}")
    
    print("\n" + "=" * 70)
    print("数据集加载完成!")
    print("=" * 70)
    
    # 返回数据加载器供后续使用
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 运行主函数
    train_loader, val_loader, test_loader = main()
    
    # 如果需要在其他脚本中使用，可以这样：
    # from load_voc_dataset import VOCDatasetConfig, create_dataloaders
    # opts = VOCDatasetConfig()
    # train_loader, val_loader, test_loader, n_classes = create_dataloaders(opts)

