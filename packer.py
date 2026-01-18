#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目代码导出脚本 - XML格式（LLM优化版）
"""

import os
from pathlib import Path
from datetime import datetime
import xml.sax.saxutils as saxutils

# ==================== 配置区域 ====================
OUTPUT_FILENAME = "project_export.xml"

# 包含的文件扩展名
INCLUDE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
    '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
    '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.md', '.txt', '.rst', '.sh', '.bash', '.sql', '.r', '.dart',
    '.gradle', '.properties', '.env', '.dockerfile'
}

# 排除的文件夹
EXCLUDE_DIRS = {
    'node_modules', 'venv', 'env', '.venv', '.env',
    '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
    'dist', 'build', 'target', 'out', 'output',
    '.idea', '.vscode', '.vs', '.settings',
    'bin', 'obj', 'pkg', 'vendor',
    '.next', '.nuxt', 'coverage', '.nyc_output',
    '.gradle', '.cache', '.mypy_cache', '.tox'
}

# 排除的文件
EXCLUDE_FILES = {
    '.DS_Store', 'Thumbs.db', 'desktop.ini',
    '.gitignore', '.gitkeep', '.dockerignore',
    OUTPUT_FILENAME,
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock'
}

# 无扩展名但需要包含的特殊文件
SPECIAL_FILES = {
    'Makefile', 'Dockerfile', 'Rakefile', 'Gemfile', 
    'Procfile', 'Vagrantfile', 'Jenkinsfile'
}


def xml_escape(text):
    """XML转义，处理特殊字符"""
    if text is None:
        return ""
    return saxutils.escape(str(text))


def should_include_file(file_path):
    """判断文件是否应该被包含"""
    file_path = Path(file_path)
    
    # 排除特定文件
    if file_path.name in EXCLUDE_FILES:
        return False
    
    # 包含特殊文件
    if file_path.name in SPECIAL_FILES:
        return True
    
    # 检查扩展名
    return file_path.suffix.lower() in INCLUDE_EXTENSIONS


def generate_tree_node(path, prefix="", is_last=True, parent_chain=None):
    """
    递归生成目录树的XML节点
    
    Args:
        path: 当前路径
        prefix: 显示用的前缀（用于树形结构）
        is_last: 是否是同级最后一个
        parent_chain: 父路径链（用于检测循环）
    
    Returns:
        (tree_lines, file_nodes) - 树形结构行列表和文件节点列表
    """
    if parent_chain is None:
        parent_chain = set()
    
    # 防止循环引用（符号链接）
    try:
        real_path = path.resolve()
        if real_path in parent_chain:
            return [], []
    except (OSError, RuntimeError):
        return [], []
    
    tree_lines = []
    file_nodes = []
    
    # 读取目录内容
    try:
        items = list(path.iterdir())
    except PermissionError:
        return [f"{prefix}[Permission Denied]"], []
    except Exception as e:
        return [f"{prefix}[Error: {str(e)}]"], []
    
    # 过滤并排序：文件夹在前，然后按名称排序
    items = [
        item for item in items
        if item.name not in EXCLUDE_DIRS and item.name not in EXCLUDE_FILES
    ]
    items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    
    # 更新父路径链
    new_chain = parent_chain | {real_path}
    
    # 遍历所有项目
    for index, item in enumerate(items):
        is_last_item = (index == len(items) - 1)
        connector = "└── " if is_last_item else "├── "
        
        if item.is_dir():
            # 处理文件夹
            tree_lines.append(f"{prefix}{connector}{item.name}/")
            
            # 递归处理子目录
            extension = "    " if is_last_item else "│   "
            sub_tree, sub_files = generate_tree_node(
                item, 
                prefix + extension, 
                is_last_item,
                new_chain
            )
            tree_lines.extend(sub_tree)
            file_nodes.extend(sub_files)
        else:
            # 处理文件
            tree_lines.append(f"{prefix}{connector}{item.name}")
            
            # 如果是需要包含的代码文件，添加到文件列表
            if should_include_file(item):
                file_nodes.append(item)
    
    return tree_lines, file_nodes


def read_file_content(file_path):
    """
    读取文件内容，尝试多种编码
    
    Returns:
        (content, encoding, error) - 文件内容、使用的编码、错误信息
    """
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                return content, encoding, None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, None, f"Read error: {str(e)}"
    
    return None, None, "Cannot decode with any known encoding"


def get_file_info(file_path, root_path):
    """获取文件的详细信息"""
    try:
        stat = file_path.stat()
        relative_path = file_path.relative_to(root_path)
        
        return {
            'path': str(relative_path),
            'name': file_path.name,
            'extension': file_path.suffix or 'none',
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception as e:
        return None


def export_project_xml(root_path=None, output_file=None):
    """
    导出项目为XML格式
    
    Args:
        root_path: 项目根路径（默认当前目录）
        output_file: 输出文件名
    """
    if root_path is None:
        root_path = os.getcwd()
    
    if output_file is None:
        output_file = OUTPUT_FILENAME
    
    root = Path(root_path).resolve()
    output_path = root / output_file
    
    print(f"=" * 60)
    print(f"项目代码导出 - XML格式")
    print(f"=" * 60)
    print(f"项目路径: {root}")
    print(f"输出文件: {output_path}")
    print(f"=" * 60)
    
    # 生成目录树和收集文件
    print("\n[1/3] 扫描项目结构...")
    tree_lines, code_files = generate_tree_node(root)
    
    # 按路径排序文件
    code_files.sort(key=lambda x: str(x.relative_to(root)).lower())
    
    print(f"✓ 找到 {len(code_files)} 个代码文件")
    
    # 统计文件类型
    stats = {}
    total_size = 0
    for file_path in code_files:
        ext = file_path.suffix or 'no_extension'
        stats[ext] = stats.get(ext, 0) + 1
        try:
            total_size += file_path.stat().st_size
        except:
            pass
    
    # 写入XML文件
    print(f"\n[2/3] 生成XML文档...")
    
    with open(output_path, 'w', encoding='utf-8') as out:
        # XML头部
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<project>\n\n')
        
        # ============ 元数据部分 ============
        out.write('  <!-- 项目元数据 -->\n')
        out.write('  <metadata>\n')
        out.write(f'    <name>{xml_escape(root.name)}</name>\n')
        out.write(f'    <path>{xml_escape(str(root))}</path>\n')
        out.write(f'    <export_time>{datetime.now().isoformat()}</export_time>\n')
        out.write(f'    <total_files>{len(code_files)}</total_files>\n')
        out.write(f'    <total_size_bytes>{total_size}</total_size_bytes>\n')
        out.write(f'    <total_size_mb>{total_size / (1024 * 1024):.2f}</total_size_mb>\n')
        out.write('    <file_types>\n')
        
        for ext, count in sorted(stats.items(), key=lambda x: (-x[1], x[0])):
            out.write(f'      <type extension="{xml_escape(ext)}" count="{count}"/>\n')
        
        out.write('    </file_types>\n')
        out.write('  </metadata>\n\n')
        
        # ============ 目录结构部分 ============
        out.write('  <!-- 目录结构 -->\n')
        out.write('  <directory_structure>\n')
        out.write('    <tree><![CDATA[\n')
        out.write(f'{root.name}/\n')
        out.write('\n'.join(tree_lines))
        out.write('\n    ]]></tree>\n')
        out.write('  </directory_structure>\n\n')
        
        # ============ 文件内容部分 ============
        out.write('  <!-- 源代码文件 -->\n')
        out.write('  <files>\n')
        
        print(f"\n[3/3] 读取文件内容...")
        
        for index, file_path in enumerate(code_files, 1):
            relative_path = file_path.relative_to(root)
            print(f"  [{index}/{len(code_files)}] {relative_path}")
            
            # 获取文件信息
            file_info = get_file_info(file_path, root)
            if not file_info:
                print(f"    ⚠ 无法获取文件信息")
                continue
            
            # 读取文件内容
            content, encoding, error = read_file_content(file_path)
            
            # 写入文件节点
            out.write(f'\n    <file index="{index}">\n')
            out.write(f'      <path>{xml_escape(file_info["path"])}</path>\n')
            out.write(f'      <name>{xml_escape(file_info["name"])}</name>\n')
            out.write(f'      <extension>{xml_escape(file_info["extension"])}</extension>\n')
            out.write(f'      <size_bytes>{file_info["size"]}</size_bytes>\n')
            out.write(f'      <modified>{file_info["modified"]}</modified>\n')
            
            if error:
                out.write(f'      <error>{xml_escape(error)}</error>\n')
                out.write('      <content/>\n')
                print(f"    ⚠ 读取失败: {error}")
            else:
                out.write(f'      <encoding>{encoding}</encoding>\n')
                out.write(f'      <lines>{len(content.splitlines())}</lines>\n')
                out.write('      <content><![CDATA[\n')
                out.write(content)
                if not content.endswith('\n'):
                    out.write('\n')
                out.write(']]></content>\n')
            
            out.write('    </file>\n')
        
        out.write('\n  </files>\n\n')
        out.write('</project>\n')
    
    # 输出统计信息
    print(f"\n{'=' * 60}")
    print(f"✓ 导出完成!")
    print(f"{'=' * 60}")
    print(f"输出文件: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"总计文件: {len(code_files)} 个")
    print(f"代码总量: {total_size / 1024:.2f} KB")
    print(f"\n文件类型统计:")
    for ext, count in sorted(stats.items(), key=lambda x: (-x[1], x[0]))[:10]:
        print(f"  {ext:15s} : {count:3d} 个")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # 导出当前目录
    export_project_xml()
    
    # 或指定其他目录
    # export_project_xml(root_path="/path/to/your/project")
    
    # 或指定输出文件名
    # export_project_xml(output_file="my_project.xml")