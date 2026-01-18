#!/bin/bash
# CUDA_CONV_CNN 环境配置脚本
# 使用方法: source setup_env.sh

set -e

ENV_NAME="cuda_conv_cnn"

echo "=========================================="
echo "  CUDA CNN 项目环境配置"
echo "=========================================="

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重建? (y/n): " choice
    if [ "$choice" = "y" ]; then
        echo "🗑️  删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "使用现有环境"
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

echo "🔧 创建 conda 环境..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "  ✅ 环境创建完成!"
echo "=========================================="
echo ""
echo "激活环境:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "编译并运行:"
echo "  ./run.sh"
echo ""
echo "单独编译:"
echo "  mkdir -p build && cd build"
echo "  cmake .."
echo "  make -j\$(nproc)"
echo ""
