#!/bin/bash

# 遇到任何错误立即停止脚本 (非常重要，防止编译失败了还硬跑 python)
set -e

echo "=========================================="
echo "      🚀 1. Cleaning Build Artifacts      "
echo "=========================================="

# 删除旧的编译文件和 .so 库
rm -rf build
# 删除当前目录下的 .so 文件 (防止 python 引用旧库)
rm -f *.so 

echo "=========================================="
echo "      🛠️  2. Configuring CMake            "
echo "=========================================="

mkdir build
cd build
cmake ..

echo "=========================================="
echo "      ⚡ 3. Compiling (Make -j)           "
echo "=========================================="

# 使用所有 CPU 核心并行编译
make -j$(nproc)

# 确保编译出的 .so 文件被复制回上级目录 (如果 CMakeLists.txt 没配自动移动)
if ls *.so 1> /dev/null 2>&1; then
    cp *.so ..
    echo " -> Copied .so to project root."
fi

cd ..

echo "=========================================="
echo "      🐍 4. Running Python Script         "
echo "=========================================="

# 默认运行 train_cifar10.py，也可以通过命令行参数传入其他文件
# 比如: ./run.sh test.py
if [ -z "$1" ]; then
    TARGET_SCRIPT="train/train_cifar10.py"
else
    TARGET_SCRIPT="$1"
fi

echo "Running: $TARGET_SCRIPT"
python "$TARGET_SCRIPT"