import os
import re
import config

def clean_and_format_txt(input_path, output_path):
    """
    通过正则匹配，清洗纯 txt 文本中的无关内容，并按一级标题分组重排
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    groups = []
    current_group = []
    
    # 停止规则：遇到参考文献直接停止读取
    stop_keywords = ['参考文献', 'references', 'reference list', 'reference']
    
    def is_garbage(text):
        # 过滤图、表、图例开头的句子（如 Fig. 1, 图 2, Table 3）
        if re.match(r'^(图|表|Fig\.|Figure|Table)\s*\d+', text, re.IGNORECASE):
            return True
        # 过滤纯数字（通常是页码）
        if re.match(r'^[\d\s]+$', text): 
            return True
        return False

    def is_level_1_heading(text):
        # 规则 1：常见的固定大标题
        if text.lower() in ['摘要', 'abstract', '结论', 'conclusion', '引言', 'introduction', '致谢', 'acknowledgements']:
            return True
        # 规则 2：数字开头且较短的句子，如 "1 地质背景", "1. Introduction" (限定长度排除带数字的长句)
        if re.match(r'^\d+[\.\s、]+[^\d]+$', text) and len(text) < 35:
            return True
        # 规则 3：如果你复制的是 MinerU 渲染后的 Markdown 文本，一级标题往往带有 #
        if text.startswith('# ') and not text.startswith('##'):
            return True
        return False

    for line in lines:
        text = line.strip()
        
        # 跳过空行
        if not text:
            continue
            
        # 清理可能带有的多余 # 号（后续由我们统一规范添加）
        clean_text = text.lstrip('#').strip()
        
        # 1. 检测是否到达参考文献区域
        if clean_text.lower() in stop_keywords or re.match(r'^\d*\.?\s*(参考文献|references)$', clean_text.lower()):
            break
            
        # 2. 过滤图表、页码等不需要的内容
        if is_garbage(text):
            continue
            
        # 3. 处理一级大标题与自然段分组
        if is_level_1_heading(text):
            if current_group:
                # 将上一个大组的内容合并
                groups.append("\n".join(current_group))
                current_group = []
            # 一级大标题单独一行，并在开头用 # 标注
            current_group.append(f"# {clean_text}")
        else:
            # 二级标题或普通正文自然段，直接放入当前组
            current_group.append(text)
            
    # 循环结束后，存入最后一组
    if current_group:
        groups.append("\n".join(current_group))
        
    # 4. 大组间用多行空行分割（\n\n\n 表示段落间空两行）
    final_text = "\n\n\n".join(groups)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

if __name__ == '__main__':
    # 初始化你的配置项
    config.init_config()
    date = config.args.date
    
    # 匹配你的输入输出路径
    origin_path = "./data/{}/text1/".format(date)
    result_path = "./data/{}/text2/".format(date)
    
    print("=" * 40)
    print(f"正在读取纯文本并进行清洗重排...")
    print(f"读取路径: {origin_path}")
    print(f"输出路径: {result_path}")
    print("=" * 40)
    
    # os.walk 自动遍历并保持多层嵌套的文件夹结构
    for root, dirs, files in os.walk(origin_path):
        for file in files:
            if file.endswith('.txt'):
                input_file = os.path.join(root, file)
                
                # 计算相对路径，同步克隆子文件夹结构
                relative_path = os.path.relpath(root, origin_path)
                out_folder = os.path.join(result_path, relative_path)
                os.makedirs(out_folder, exist_ok=True)
                
                output_file = os.path.join(out_folder, file)
                clean_and_format_txt(input_file, output_file)
                print(f"已清洗并重排: {os.path.join(relative_path, file)}")
                
    print("\n所有文本处理完成！")