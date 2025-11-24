# PyHiVision

<div align="center">

![pyhivision](https://socialify.git.ci/ZyphrZero/pyhivision/image?description=1&font=Jost&issues=1&language=1&name=1&owner=1&pattern=Signal&stargazers=1&theme=Dark)

<h3>PyHiVision 证件照处理 SDK</h3>

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](https://github.com/ZyphrZero/pyhivision/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[快速开始](#-快速开始) • [配置说明](#️-配置说明) • [API 文档](#-api-文档) • [贡献指南](#-贡献指南)

</div>

---

## 📖 简介

PyHiVision 是一个专业级的证件照处理 SDK，集成多种先进的 AI 模型，为证件照制作、人像处理和批量照片编辑提供完整的解决方案。

<table>
<tr>
<td width="50%">

### ✨ 核心特性

- 🚀 **高性能架构** - 简洁高效的同步处理管线
- 🤖 **多模型支持** - ModNet、BiRefNet、RMBG、MTCNN、RetinaFace
- 🎨 **完整流程** - 抠图 → 检测 → 美颜 → 调整 → 背景
- 💾 **智能管理** - LRU 缓存，自动内存管理
- ⚡ **GPU 加速** - CUDA 加速，线程池优化
- 🔧 **灵活配置** - 环境变量、配置文件、代码配置
- 🛡️ **类型安全** - 完整类型注解与数据验证

</td>
<td width="50%">

### 🎯 适用场景

- 📸 证件照在线制作平台
- 🖼️ 批量人像照片处理系统
- 👤 人脸识别与美颜应用
- 🤖 图像自动化处理服务
- 🎨 AI 驱动的图像编辑工具

</td>
</tr>
</table>

---

## 🚀 快速开始

### 安装

```bash
# 基础安装
pip install pyhivision

# 开发环境（包含测试和代码检查工具）
pip install "pyhivision[dev]"

# GPU 加速版本
pip install "pyhivision[gpu]"
```

### 基本使用

```python
import cv2
from pyhivision import IDPhotoSDK, PhotoRequest, create_settings

def main():
    # 配置模型路径
    settings = create_settings(
        matting_models_dir="~/.pyhivision/models/matting",
        detection_models_dir="~/.pyhivision/models/detection",  # MTCNN 不需要
    )

    # 创建 SDK 实例
    sdk = IDPhotoSDK.create(settings=settings)

    # 读取图像
    image = cv2.imread("input.jpg")

    # 创建请求
    request = PhotoRequest(
        image=image,
        size=(413, 295),  # 一寸照尺寸
        background_color=(255, 0, 0),  # 蓝色背景
        matting_model="modnet_photographic",
        detection_model="mtcnn"
    )

    # 处理图像
    result = sdk.process_single(request)

    # 保存结果
    cv2.imwrite("standard.jpg", result.standard)
    if result.hd is not None:
        cv2.imwrite("hd.jpg", result.hd)

    print(f"✅ 处理完成")

if __name__ == "__main__":
    main()
```

---

## ⚙️ 配置说明

### 代码配置（推荐）

```python
from pyhivision import create_settings

# 基础配置
settings = create_settings(
    # 模型路径（必需）
    matting_models_dir="~/.pyhivision/models/matting",      # 抠图模型目录
    detection_models_dir="~/.pyhivision/models/detection",  # 检测模型目录（MTCNN 除外）

    # 性能配置
    enable_gpu=False,          # 是否启用 GPU
    num_threads=4,             # ONNX Runtime 线程数
    model_cache_size=3,        # 模型缓存数量

    # 日志配置
    log_level="INFO",          # 日志级别
)
```

### 环境变量配置

```bash
# 模型路径（必需）
export HIVISION_MATTING_MODELS_DIR=~/.pyhivision/models/matting
export HIVISION_DETECTION_MODELS_DIR=~/.pyhivision/models/detection

# 性能配置
export HIVISION_ENABLE_GPU=true
export HIVISION_NUM_THREADS=8

# 日志配置
export HIVISION_LOG_LEVEL=DEBUG
```

### 配置优先级

1. 代码中直接传参：`create_settings(enable_gpu=True)`
2. 环境变量：`HIVISION_ENABLE_GPU=true`
3. `.env` 文件：`HIVISION_ENABLE_GPU=true`
4. 默认值

### 主要配置项

<table>
<thead>
<tr>
<th width="25%">配置项</th>
<th width="20%">类型</th>
<th width="15%">默认值</th>
<th width="40%">说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>matting_models_dir</code></td>
<td>Path/str/None</td>
<td>None</td>
<td><strong>⚠️ 抠图模型目录（必需）</strong></td>
</tr>
<tr>
<td><code>detection_models_dir</code></td>
<td>Path/str/None</td>
<td>None</td>
<td>检测模型目录（MTCNN 除外）</td>
</tr>
<tr>
<td><code>enable_gpu</code></td>
<td>bool</td>
<td>False</td>
<td>是否启用 GPU 加速</td>
</tr>
<tr>
<td><code>num_threads</code></td>
<td>int</td>
<td>4</td>
<td>ONNX Runtime 线程数</td>
</tr>
<tr>
<td><code>model_cache_size</code></td>
<td>int</td>
<td>3</td>
<td>模型缓存数量上限</td>
</tr>
<tr>
<td><code>max_image_size</code></td>
<td>int</td>
<td>2000</td>
<td>图像最大边长</td>
</tr>
<tr>
<td><code>log_level</code></td>
<td>str</td>
<td>"INFO"</td>
<td>日志级别</td>
</tr>
</tbody>
</table>

> 💡 **提示**：模型路径由上层应用控制，SDK 不提供默认路径。推荐使用用户目录：`~/.pyhivision/models/`

---

## 🏗️ 架构概览

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                     用户层 (User Layer)                      │
│         IDPhotoSDK.process() / PhotoPipeline                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   核心管线 (Core Pipeline)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 抠图 (Matting) → ModNet/BiRefNet/RMBG          │   │
│  │ 2. 检测 (Detection) → MTCNN/RetinaFace            │   │
│  │ 3. 美颜 (Beauty) → 亮度/对比度/磨皮/美白           │   │
│  │ 4. 调整 (Adjustment) → 裁剪/缩放/布局              │   │
│  │ 5. 背景 (Background) → 纯色/模板                   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 底层服务 (Infrastructure)                    │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ ModelManager │  │ ResultCache  │                        │
│  │ (模型管理)    │  │ (结果缓存)    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 模块结构

```
pyhivision/
├── config/          # 配置管理（Pydantic Settings）
├── core/            # 核心流程编排（Pipeline、ModelManager）
├── models/          # AI 模型封装（Detection、Matting）
├── processors/      # 图像处理器（抠图、检测、美颜、布局）
├── schemas/         # 数据模型（Request、Response、Validation）
├── utils/           # 工具函数（Logger、Image、Compression）
├── exceptions/      # 异常定义（自定义错误类型）
└── assets/          # 资源文件（LUT、字体、模板）
```

---

## 🛠️ 开发指南

### 开发环境搭建

```bash
# 克隆仓库
git clone https://github.com/ZyphrZero/pyhivision.git
cd pyhivision

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

### 代码规范

- **格式化工具**: Ruff（替代 Black + isort）
- **行长度**: 100 字符
- **Python 版本**: 3.11-3.12
- **类型注解**: 必须使用完整的类型提示

```bash
# 代码格式检查
ruff check .

# 自动修复
ruff check --fix .
```

### 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行覆盖率测试
pytest tests/ --cov=pyhivision --cov-report=html

# 运行特定测试
pytest tests/test_pipeline.py -v
```

### 编码规范

- **类名**: PascalCase（如 `ModelManager`）
- **函数/变量**: snake_case（如 `load_model`, `face_info`）
- **常量**: UPPER_SNAKE_CASE（如 `CUBE64_SIZE`）
- **私有方法**: 前缀 `_`（如 `_create_session`）
- **错误处理**: 使用自定义异常类（继承自 `HivisionError`）

---

## 📚 API 文档

### 核心类

#### `IDPhotoSDK`

主要的 SDK 接口，提供证件照处理功能。

```python
sdk = IDPhotoSDK.create(settings=settings)
result = sdk.process_single(request)
```

#### `PhotoRequest`

照片处理请求对象。

```python
request = PhotoRequest(
    image=image,                          # np.ndarray
    size=(413, 295),                      # 输出尺寸
    background_color=(255, 0, 0),         # 背景色（BGR）
    matting_model="modnet_photographic",  # 抠图模型
    detection_model="mtcnn",              # 检测模型
    beauty_params=BeautyParams(           # 美颜参数（可选）
        brightness=10,
        contrast=5
    )
)
```

#### `PhotoResult`

照片处理结果对象。

```python
result.standard         # 标准照片（np.ndarray）
result.hd              # 高清照片（可选）
result.face_info       # 人脸信息
```

### 支持的模型

<table>
<tr>
<td width="50%">

#### 🎨 抠图模型

| 模型名称 | 说明 |
|---------|------|
| `modnet_photographic` | 通用摄影抠图（官方权重） |
| `hivision_modnet` | HiVision 优化版 |
| `birefnet_lite` | BiRefNet 轻量版 |
| `rmbg_1_4` | RMBG 1.4 版本 |

</td>
<td width="50%">

#### 👤 人脸检测模型

| 模型名称 | 说明 |
|---------|------|
| `mtcnn` | MTCNN（内置权重） ✅ |
| `retinaface` | RetinaFace（需配置路径） |

</td>
</tr>
</table>

---

## 🤝 贡献指南

欢迎贡献代码、报告 Bug 或提出新功能建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献要求

- 遵循项目的编码规范（Ruff）
- 添加必要的单元测试
- 更新相关文档
- 确保所有测试通过

---

## 🙏 致谢

- [HiVision 证件照项目](https://github.com/Zeyi-Lin/HivisionIDPhotos) - HiVision 证件照项目
- [ModNet](https://github.com/ZHKKKe/MODNet) - 高性能人像抠图模型
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - 高精度背景移除模型
- [MTCNN](https://github.com/ipazc/mtcnn) - 多任务级联卷积神经网络人脸检测
- [RetinaFace](https://github.com/serengil/retinaface) - 先进的人脸检测模型

---

## 📧 联系方式

- **作者**: FastParse Team
- **问题反馈**: [GitHub Issues](https://github.com/ZyphrZero/pyhivision/issues)

---

<div align="center">

**用 ❤️ 构建 | Made with Love**

⭐ 如果这个项目对你有帮助，请给我们一个 Star！

</div>

---

## 📄 许可证

本项目采用 **Apache License 2.0** 开源协议。
