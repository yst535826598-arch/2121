import os
import config  # 引入你的配置文件

def batch_convert_nested_texts(input_base_dir, output_base_dir):
    """
    遍历多层嵌套文件夹，将普通txt转换为抽取代码所需的单行字典格式
    """
    if not os.path.exists(input_base_dir):
        print(f"错误：找不到输入文件夹路径 {input_base_dir}")
        return

    # os.walk 能够自动穿透多层文件夹（例如：变质火山-沉积岩型\草店）
    for root, dirs, files in os.walk(input_base_dir):
        for filename in files:
            if not filename.endswith('.txt'):
                continue
                
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_base_dir)
            output_dir = os.path.join(output_base_dir, relative_path)
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # --- 读取并切分文本 ---
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            raw_paragraphs = content.split('\n')
            chunks = []
            for para in raw_paragraphs:
                para = para.strip()
                if not para:
                    continue
                # 按句号切分，但保留句号
                sentences = para.split('。')
                for i, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if i < len(sentences) - 1:
                        chunks.append(sentence + '。')
                    else:
                        chunks.append(sentence)
                        
            # --- 构建目标字典结构 ---
            result_dict = {}
            text_index = 0
            for chunk in chunks:
                if len(chunk) < 5: 
                    continue
                    
                result_dict[f'text_{text_index}'] = {
                    'label': '正文', 
                    'text': chunk
                }
                text_index += 1
                
            # --- 写入新文件（单行字符串形式） ---
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(str(result_dict))
                
            print(f"已转换: {os.path.join(relative_path, filename)}")

if __name__ == '__main__':
    # 初始化配置
    config.init_config()
    
    # 你的基础参数获取
    date = config.args.date
    figure_array = config.args.KGfigure_labels
    entitys_labels_dict = config.args.KGentity_labels
    
    # 原始文本读取路径
    origin_path = "./data/{}/text2/".format(date)
    
    # 转换后的文本输出路径（建议用一个新文件夹，比如 formatted_origin）
    formatted_path = "./data/{}/origin/".format(date)
    
    print(f"===== 开始执行文本预处理 =====")
    print(f"输入路径: {origin_path}")
    print(f"输出路径: {formatted_path}")
    
    # 执行批量转换
    batch_convert_nested_texts(origin_path, formatted_path)
    
    print(f"===== 转换完成 =====")