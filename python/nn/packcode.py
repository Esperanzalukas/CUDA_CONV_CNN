import os

# ================= 配置区域 =================

# 输出文件的名称
OUTPUT_FILE = "project_code_summary.txt"

# 需要忽略的文件夹 (完全匹配)
IGNORE_DIRS = {
    "build",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "data",       # 数据集通常很大，不打包
    "logs",       # 日志文件不打包
    "dist",
    "egg-info"
}

# 需要忽略的文件后缀 (以这些结尾的文件会被忽略)
IGNORE_EXTENSIONS = {
    ".so",        # 编译出的动态库
    ".o",         # 目标文件
    ".pyc",       # Python 字节码
    ".pyd",       # Windows Python 扩展
    ".exe",
    ".bin",
    ".pkl",       # 模型权重或 pickle 文件
    ".pth",       # PyTorch 权重
    ".jpg", ".png", ".jpeg", # 图片
    ".zip", ".tar", ".gz",   # 压缩包
    ".pdf",       # 文档
    ".DS_Store"   # Mac 系统文件
}

# 明确包含的文件后缀 (只打包这些，或者为空则打包除了忽略以外的所有)
# 如果只想打包代码，建议设置如下：
INCLUDE_EXTENSIONS = {
    ".py",
    ".cu",        # CUDA 源码
    ".cc", ".cpp", ".c", ".h", ".hpp", # C++ 源码
    ".sh",        # Shell 脚本
    ".txt", ".md", # 文档
    "Makefile",
    "CMakeLists.txt"
}
# 如果设为 None，则打包除了 IGNORE_EXTENSIONS 以外的所有文件
# INCLUDE_EXTENSIONS = None 

# ===========================================

def is_ignored(path, filename):
    # 1. 检查是否在忽略的文件夹中
    parts = path.split(os.sep)
    for part in parts:
        if part in IGNORE_DIRS:
            return True
            
    # 2. 检查文件后缀是否在忽略列表中
    _, ext = os.path.splitext(filename)
    if ext.lower() in IGNORE_EXTENSIONS:
        return True
    
    # 3. 检查是否在包含列表中 (如果设置了包含列表)
    if INCLUDE_EXTENSIONS is not None:
        if ext.lower() not in INCLUDE_EXTENSIONS and filename not in INCLUDE_EXTENSIONS:
            return True

    # 4. 忽略输出文件本身和脚本本身
    if filename == OUTPUT_FILE or filename == os.path.basename(__file__):
        return True
        
    return False

def pack_code():
    cwd = os.getcwd()
    print(f"📦 开始打包目录: {cwd}")
    
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # 写入头部信息
        outfile.write(f"Project Code Summary\n")
        outfile.write(f"Generated from: {cwd}\n")
        outfile.write("="*50 + "\n\n")

        for root, dirs, files in os.walk(cwd):
            # 过滤掉忽略的目录，修改 dirs 列表会影响 os.walk 的后续遍历
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                # 获取相对路径
                rel_path = os.path.relpath(file_path, cwd)
                
                if not is_ignored(rel_path, file):
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                            content = infile.read()
                            
                        # 写入文件分隔符和内容
                        outfile.write(f"\n{'='*20} START OF FILE: {rel_path} {'='*20}\n")
                        outfile.write(content)
                        outfile.write(f"\n{'='*20} END OF FILE: {rel_path} {'='*20}\n")
                        print(f"✅ 添加: {rel_path}")
                        count += 1
                    except Exception as e:
                        print(f"❌ 读取错误 {rel_path}: {e}")

    print(f"\n🎉 打包完成！共处理 {count} 个文件。")
    print(f"📁 结果保存在: {os.path.join(cwd, OUTPUT_FILE)}")

if __name__ == "__main__":
    pack_code()