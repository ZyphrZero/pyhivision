# PyHiVision 示例代码

本目录包含 PyHiVision SDK 的各类使用示例。

## 📋 示例列表

| 示例 | 文件 | 说明 |
|------|------|------|
| **基础使用** | `01_basic_usage.py` | 最简单的证件照生成示例 |
| **美颜增强** | `02_beauty_enhancement.py` | 使用美颜参数优化照片 |
| **批量处理** | `03_batch_processing.py` | 批量处理多张照片 |
| **自定义布局** | `04_custom_layout.py` | 调整人脸位置和布局 |
| **多尺寸生成** | `05_different_sizes.py` | 生成不同规格的证件照 |
| **错误处理** | `06_error_handling.py` | 完整的异常处理示例 |
| **GPU 加速** | `07_gpu_acceleration.py` | 使用 GPU 加速处理 |
| **仅换背景** | `08_change_background_only.py` | 跳过人脸检测，仅更换背景 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

以可编辑模式安装能够保持与正式发布版本一致的导入路径，示例中的任何更改也不会影响到全局 Python 解释器。

### 2. 下载模型

确保已下载所需的模型文件到指定目录：

```
~/.pyhivision/models/
├── matting/
│   ├── modnet_photographic_portrait_matting.onnx
│   └── ...
└── detection/
    └── ...（MTCNN 无需额外文件）
```

### 3. 运行示例

```bash
# 基础使用
python examples/01_basic_usage.py

# 美颜增强
python examples/02_beauty_enhancement.py

# 批量处理
python examples/03_batch_processing.py
```

## 📝 常见证件照尺寸

| 规格 | 尺寸（像素） | 用途 |
|------|-------------|------|
| 一寸 | 413 × 295 | 简历、档案 |
| 二寸 | 626 × 413 | 毕业证、工作证 |
| 小一寸 | 390 × 260 | 驾驶证、身份证 |
| 护照 | 354 × 472 | 护照、签证 |

## 🎨 背景颜色参考

```python
# BGR 格式
background_color = (255, 255, 255)  # 白色
background_color = (255, 0, 0)      # 蓝色
background_color = (0, 0, 255)      # 红色
background_color = (0, 255, 0)      # 绿色
```

## ⚙️ 配置说明

### 基础配置

```python
from pyhivision import create_settings

settings = create_settings(
    matting_models_dir="~/.pyhivision/models/matting",
    detection_models_dir="~/.pyhivision/models/detection",
    enable_gpu=False,
    num_threads=4,
)
```

### 美颜参数

```python
from pyhivision import BeautyParams

beauty = BeautyParams(
    brightness=10,      # 亮度 (-100~100)
    contrast=5,         # 对比度 (-100~100)
    whitening=15,       # 美白 (0~30)
    skin_smoothing=5,   # 磨皮 (0~10)
)
```

### 布局参数

```python
from pyhivision import LayoutParams

layout = LayoutParams(
    head_measure_ratio=0.2,   # 头部宽度比例 (0.1~0.5)
    head_height_ratio=0.45,   # 头顶高度比例 (0.3~0.7)
)
```

## 🔧 故障排除

### 问题：找不到模型文件

**解决方案**：确认模型文件路径正确，并且文件已下载。

```python
settings = create_settings(
    matting_models_dir="/path/to/your/models/matting",
)
```

### 问题：未检测到人脸

**解决方案**：
1. 确保照片中有人脸且足够清晰
2. 尝试使用不同的检测模型
3. 使用 `change_bg_only=True` 跳过人脸检测

### 问题：处理速度慢

**解决方案**：
1. 启用 GPU 加速（需要安装 `onnxruntime-gpu`）
2. 增加线程数 `num_threads=8`
3. 使用更轻量的模型

## 📚 更多资源

- [GitHub 仓库](https://github.com/ZyphrZero/pyhivision)
- [API 文档](../README.md)
- [问题反馈](https://github.com/ZyphrZero/pyhivision/issues)
