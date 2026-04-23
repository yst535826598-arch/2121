import os
import shutil  
from concurrent.futures import ThreadPoolExecutor, as_completed # 💡 新增：多线程库
from utils.LLM_function import *
import config
config.init_config()

def move_file(src_file, dst_folder):
    shutil.move(src_file, dst_folder)

def step1_read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        texts=file.readline()
        text_dict_origin=eval(texts)
    text_dict={}
    id=''
    for id in text_dict_origin:
        if len(text_dict_origin[id]['text'])<10:
            print(text_dict_origin[id]['text'])
            print(1)
            continue
        text_dict[id]={}
        # 【换壳】：探测区 -> 具体技术和算法模型
        text_dict[id]['具体技术和算法模型']=tech_model_name
        text_dict[id]['文本']=text_dict_origin[id]['text']
        text_dict[id]['抽取的实体']={}
        text_dict[id]['抽取的三元组']=[]
    return text_dict

def step2_NER_NC_RC(my_text,entitys_labels_dict):
    final={}
    # 【重构：大幅降低成本】关闭 repetitions=2（重复抽取），单次抽取准确率已经足够！
    entity,figure=level1_entity_multiple_strategy(my_text,strategy='multiple',repetitions=1)
    print("获得如下地质概念:"+str(entity))
    if entity=="ERROR" or len(entity)<1:
        return "ERROR","ERROR","ERROR"
        
    print("执行批量划分标签")
    entitys_labels_array=[]
    for label in entitys_labels_dict.keys():
        entitys_labels_array.append(label)
    
    # ==== 【重构核心：增加强力实体噪音过滤器 (解决问题三)】 ====
    valid_entities = []
    # 💡 定义图谱停用词（黑名单）：这些词太宽泛，单独作为节点毫无意义
    stop_words = ['卫星', '算法', '模型', '数据', '方法', '公式', '参数', '变量', '系统', '网络', '技术', '理论', '特征']
    # 💡 定义数学公式常见的特殊非法字符
    math_symbols = ['=', '+', '{', '}', 'ₖ', 'ₙ', 'ˡ', 'ω', '∑', '∫', 'α', 'β', 'γ', 'θ', 'λ', 'μ']
    
    for two in entity:
        two = str(two).strip()
        # 1. 基础过滤
        if two == 'ERROR' or two == '无#无' or '具体技术' in two or '算法模型' in two:
            continue
        # 2. 过滤单字母/单字符（秒杀 'E', 'O' 等）
        if len(two) <= 1:
            print(f"🛡️ 拦截单字符噪音: 【{two}】")
            continue
        # 3. 过滤黑名单孤立词（秒杀 '卫星', '算法' 等）
        if two in stop_words:
            print(f"🛡️ 拦截宽泛孤立词: 【{two}】")
            continue
        # 4. 过滤包含上下标或特殊符号的数学公式
        if any(sym in two for sym in math_symbols):
            print(f"🛡️ 拦截数学公式噪音: 【{two}】")
            continue
            
        valid_entities.append(two)
            
    if len(valid_entities) > 0:
        get_label = level1_entity_label_batch(valid_entities, my_text, entitys_labels_array)
    else:
        get_label = {}
        
    if get_label == "ERROR":
        get_label = {}
        
    print("获得如下地质概念的标签:"+str(get_label))
    
    # 处理数值/公式，同样采用批量模式
    if len(figure)<1 or figure=="ERROR" or figure==['无#无']:
        figure_label={}
        print("此片段无数值型信息")
    else:
        print("执行数值标签批量划分")
        # 把 figure 中的 meaning 部分提取出来，批量发给大模型
        meanings = []
        for two in figure:
            if '#' in two:
                meanings.append(two.split('#')[0])
            else:
                meanings.append(two)
                
        figure_label = {}
        if len(meanings) > 0:
             batch_result = level1_entity_label_batch(meanings, my_text, config.args.KGfigure_labels)
             if batch_result != "ERROR" and isinstance(batch_result, dict):
                 # 💡 核心修复：把大模型分好的类，重新映射回带有 '#' 的完整字符串上！
                 for two in figure:
                     meaning_key = two.split('#')[0] if '#' in two else two
                     if meaning_key in batch_result:
                         figure_label[two] = batch_result[meaning_key]

    print("获得如下数值的标签:"+str(figure_label))
    
    # 结果拼装
    for un in get_label.keys():
        label=get_label[un]
        # 【换壳】：探测区/工区 -> 具体技术和算法模型/算法模型
        if label.find('具体技术和算法模型')!=-1 or label.find('算法模型')!=-1:
            print("这是个核心技术/算法模型，略过")
            continue
        
        # 精确匹配你的 config 词典
        answer = '无'
        for c in entitys_labels_dict.keys():
            if label.find(c)!=-1 and c!='数值与公式':
                answer = c
                break
        
        if answer == '无':
            answer = label # 没匹配上，用大模型的原词
            
        final[un] = answer

    # 三元组抽取
    my_ex_time=0
    triple_2=[]
    triple_1="ERROR"
    while(my_ex_time<2):
        if my_ex_time==0:
            mtime=0
            while(triple_1=="ERROR"and mtime<2):
                mtime=mtime+1
                triple_1=level2_relation_extract(entity,my_text)
                if type(triple_1) is not list:
                    print('{}次'.format(str(mtime)))
                    triple_1="ERROR"
                    continue
            my_ex_time=my_ex_time+1
            continue
        if triple_1=="ERROR":
            triple_2=[]
            break
        for i in triple_1:
            sp=str(i).split('#')
            try:
                head=sp[0]
                tail=sp[1]
                relation=sp[2]
            except:
                continue 
            if final.get(head,-100)!=-100 and final.get(tail,-100)!=-100:        
                triple_2.append(i)
        my_ex_time=my_ex_time+1
        break
    return final,triple_2,figure_label

def step3_categorize(text_dict2):
    unite={}
    full_word={}
    merge_history={}
    mai_entity_list={}
    figure_list={}
    figure_array=config.args.KGfigure_labels
    for labeli in figure_array:
        figure_list[labeli]=[]
    for i in text_dict2.keys():
        unite[i]=text_dict2[i]['抽取的实体']
        t_figure_list=text_dict2[i]['抽取的数值']
        for name in t_figure_list.keys():
            if figure_list.get(t_figure_list[name],-100)==-100 and t_figure_list[name]!="其他数值或符号":
                figure_list[t_figure_list[name]]=[]
                # 【换壳】：area_name -> tech_model_name
                figure_dict={'内容':name,'来源':tech_model_name+'#'+files+'#'+i}
                figure_list[t_figure_list[name]].append(figure_dict)
            else:
                figure_dict={'内容':name,'来源':tech_model_name+'#'+files+'#'+i}
                figure_list[t_figure_list[name]].append(figure_dict)

    for i in unite.keys():
        t=unite[i]
        for l in t.keys():
            mai_entity_list[l]=t[l]
            label=t[l]
            if temp_label.get(label,-100)==-100:
                temp_label[label]={}
                temp_label[label][l]=[i]
                continue
            if temp_label[label].get(l,-100)==-100:
                temp_label[label][l]=[i]
            else:
                temp_label[label][l].append(i)
    temp_tri=[]
    for text in text_dict2.keys():
        triplet=text_dict2[text]['抽取的三元组']
        for tri in triplet:
            sp=str(tri).split('#')
            try:
                head=sp[0]
                tail=sp[1]
                relation=sp[2]
            except:
                continue
            if merge_history.get(head,-100)!=-100:
                head=merge_history[head]
            if merge_history.get(tail,-100)!=-100:
                tail=merge_history[tail]
            temp_tri.append({
                'head':head,
                'tail':tail,
                'relation':relation,
                'textID':text
                })
            full_word[head]=1
            full_word[tail]=1

    temp_label['全部三元组']=temp_tri
    final_word_array_addition={}
    final_word_array_entity={}
    for word in full_word.keys():            
        if mai_entity_list.get(word,-100)!=-100:
            final_word_array_entity[word]={'类型': '不明', '唯一性': '不明','标签':mai_entity_list[word]}
        else:
            # 【换壳】：非探测区属性实例 -> 非技术模型属性实例
            final_word_array_addition[word]={'类型': '不明', '唯一性': '不明','标签':'非技术模型属性实例'}

    return final_word_array_entity,final_word_array_addition,figure_list

# 💡 新增：单文本块处理封装函数（为了配合多线程）
def process_single_chunk(text_id, my_text, entitys_labels_dict):
    try:
        entitys, triples, figures = step2_NER_NC_RC(my_text, entitys_labels_dict)
        if entitys == "ERROR":
            return text_id, None
        return text_id, {
            '抽取的实体': entitys,
            '抽取的三元组': triples,
            '抽取的数值': figures
        }
    except Exception as e:
        print(f"❌ 块 {text_id} 抽取异常: {str(e)}")
        return text_id, None


if __name__ == '__main__':
    date=config.args.date
    origin_path="./data/{}/origin/".format(date)
    result_path="./data/{}/step1_result/".format(date)
    figure_array=config.args.KGfigure_labels
    entitys_labels_dict=config.args.KGentity_labels
    mission=os.listdir(origin_path)
    
    # 💡 设置并发线程数：建议保持 5-8 左右。
    # 设得太高可能会触发大模型 API 的并发限制 (Rate Limit) 报错。
    MAX_WORKERS = 8 
    
    # 【换壳】：area_name -> tech_model_name
    for tech_model_name in mission:
        result_path1=os.path.join(result_path,tech_model_name)
        tech_files_path=os.path.join(origin_path,tech_model_name)
        folder=os.listdir(tech_files_path)
        for files in folder:
            print(tech_model_name)
            print(files)
            save={}
            temp_label={}
            for label in entitys_labels_dict.keys():
                if label=='数值与公式':
                    continue
                temp_label[label]={}
            target_file_path=os.path.join(tech_files_path,files)
            if os.path.exists(os.path.join(result_path1,'{}抽取结果#'.format(date)+files)) is True:
                print("略过一个已抽取完成文件")
                continue
            else:
                os.makedirs(result_path1, exist_ok=True)
            text_dict=step1_read_file(target_file_path)                  
            print(f"✅ 完成读取，文献块数量：{len(text_dict)}。准备启动多线程并发提取...")
            
            # ========================================================
            # 💡 核心提速改造：引入 ThreadPoolExecutor 多线程并发处理
            # ========================================================
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # 提交所有段落的任务到线程池
                futures = {executor.submit(process_single_chunk, tid, text_dict[tid]['文本'], entitys_labels_dict): tid for tid in text_dict.keys()}
                
                # 等待并获取结果
                for future in as_completed(futures):
                    tid = futures[future]
                    result_id, result_data = future.result()
                    if result_data is not None:
                        save[result_id] = result_data
            # ========================================================
            
            print("🎉 所有文本块并发提取完成！进行归类...")
            text_dict2=eval(str(save))        
            final_word_array_entity,final_word_array_addition,figure_list=step3_categorize(text_dict2)
            temp_label['实例词表']=final_word_array_entity
            temp_label['扩展词表']=final_word_array_addition
            temp_label['数值表']=figure_list
            with open(os.path.join(result_path1,'{}抽取结果#'.format(date)+files), "w",encoding='utf-8') as f:
                f.write(str(temp_label))