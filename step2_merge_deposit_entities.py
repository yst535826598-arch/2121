import os
import chromadb
import numpy as np  # 💡 向量余弦相似度计算
from utils.LLM_function import *
from utils.Vector_Database_function import *
import config
config.init_config()

def load2chromadb(origin,temp_save_dict,collection_entity,full_word_table_entity):
    entitys_biaoqian={}
    for label in origin.keys():
        if label=="实例词表" or label=="全部三元组" or label=="数值表" or temp_save_dict.get(label,-100)==-100:
            continue
        for item in origin[label]:
            try:
                t_save=origin[label][item][0]
            except:
                t_save=origin[label][item]
            
            if temp_save_dict[label].get(item,-100)==-100:
                temp_save_dict[label][item]={}
                entitys_biaoqian[item]=label
            else:
                entitys_biaoqian[item]=label
    word_table_entity=origin["实例词表"]
    for word in word_table_entity.keys():
        flag=0
        t_label='不明'
        if full_word_table_entity.get(word,-100)==-100:
            try:
                t_label=word_table_entity[word]['类型']
                full_word_table_entity[word]=word_table_entity[word]
            except:
                flag=1
                full_word_table_entity[word]={'类型': '', '唯一性': '否', '标签': ''}
            if entitys_biaoqian.get(word,-100)==-100:
                del full_word_table_entity[word]
                continue
            full_word_table_entity[word]['标签']=entitys_biaoqian[word]
            emb=my_embeddings_fuction(word)
            full_word_table_entity[word]['嵌入向量']=emb
            if t_label=='不明' or flag==1:
                full_word_table_entity[word]['类型']="不明"
                full_word_table_entity[word]['唯一性']="不明"
                t_label="不明"          
            add_attribute_to_chromadb(collection_entity,word,emb)
    return full_word_table_entity,temp_save_dict,collection_entity

# 【换壳】：area_path -> tech_model_path
def step1_load(tech_model_path, collection_entity):
    full_word_table_entity = {}
    temp_save_dict = {}
    figure_dict = {}
    full_triplet = []
    figure_label = config.args.KGfigure_labels
    entitys_labels_dict = config.args.KGentity_labels
    
    for label in figure_label:
        figure_dict[label] = []
    for label in entitys_labels_dict.keys():
        temp_save_dict[label] = {}

    # =========================================================
    # 💡 【向量动态对齐核心】：预先计算所有标准标签的 Embedding
    # =========================================================
    def cosine_similarity(v1, v2):
        vec1 = np.array(v1)
        vec2 = np.array(v2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0: return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
    print("\n⏳ 正在预计算标准数值标签的向量，启动幻觉标签动态对齐...")
    standard_figure_embs = {}
    for label in figure_label:
        standard_figure_embs[label] = my_embeddings_fuction(label)
    print("✅ 标准标签向量库加载完毕！")
    # =========================================================

    file_number = 0
    
    # --- 核心修改：使用 os.walk 自动遍历所有子文件夹 ---
    for root, dirs, files in os.walk(tech_model_path):
        for filename in files:
            # 过滤掉隐藏文件
            if filename.startswith('.'):
                continue
                
            # 拼接得到文件的完整路径
            file_path = os.path.join(root, filename)
            print(f"读取到文件: {file_path}")
            
            file_number += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts = file.readline()
                    if not texts.strip(): # 检查是否为空行
                        continue
                    origin_dict = eval(texts)
                
                # 💡【替换】：处理数值表，开启向量动态对齐
                for flabel in list(origin_dict.get('数值表', {}).keys()):
                    target_label = flabel
                    
                    # 如果遇到了没见过的大模型幻觉标签
                    if target_label not in figure_label:
                        try:
                            # 1. 计算幻觉标签的向量
                            wild_emb = my_embeddings_fuction(target_label)
                            best_label = None
                            max_sim = -1.0
                            
                            # 2. 与所有标准标签库对比
                            for std_label, std_emb in standard_figure_embs.items():
                                sim = cosine_similarity(wild_emb, std_emb)
                                if sim > max_sim:
                                    max_sim = sim
                                    best_label = std_label
                            
                            # 3. 设定阈值（0.8），超过则强制纠正归位，否则丢弃
                            if max_sim >= 0.8:
                                print(f"🔄 向量对齐成功: 幻觉标签【{target_label}】被纠正为【{best_label}】 (相似度: {max_sim:.2f})")
                                target_label = best_label
                            else:
                                print(f"⚠️ 拦截失败标签: 【{target_label}】与所有标准标签相似度极低 (最高仅 {max_sim:.2f})，予以丢弃。")
                                continue
                                
                        except Exception as e:
                            print(f"⚠️ 向量计算异常: {e}，跳过标签【{target_label}】")
                            continue
                    
                    # 此时的 target_label 已经是 100% 安全合法的了！
                    for i in origin_dict['数值表'][flabel]:
                        figure_dict[target_label].append(i)
                
                # 处理三元组
                triplet = origin_dict.get("全部三元组", [])
                for tri in triplet:
                    temp_tri = eval(str(tri))
                    temp_tri['article'] = filename
                    full_triplet.append(temp_tri)
                
                # 同步到向量数据库
                full_word_table_entity, temp_save_dict, collection_entity = load2chromadb(
                    origin_dict, temp_save_dict, collection_entity, full_word_table_entity
                )
            except Exception as e:
                print(f"警告：处理文件 {filename} 时出错: {e}")
                continue

    return full_word_table_entity, full_triplet, temp_save_dict, figure_dict, file_number, collection_entity

def step2_merge(word_table_entity_mirror,full_triplet,collection_entity):
    merge_history={}
    for word in word_table_entity_mirror.keys():
        if merge_history.get(word,-100)!=-100:
            continue
        emb=word_table_entity_mirror[word]['嵌入向量']
        similar,similarID,similar_distance=determine_attribute_distance(collection_entity,emb,10)
        mapping={}
        array_4_aligen=[]
        for code in range(0,len(similar)):
            distance=similar_distance[code]
            if distance<global_distance:
                array_4_aligen.append(similar[code])
                mapping[similar[code]]={'ID':similarID[code],
                                        '嵌入向量':word_table_entity_mirror[similar[code]]['嵌入向量'],
                                        '距离':similar_distance[code]
                                        }
        united=level2_merge_special(array_4_aligen)
        if str(united).find('NO')!=-1 or str(united).find('ERROR')!=-1:
            tt_result=[]
            for code in array_4_aligen:
                if code==word:
                    continue
                else:
                    tt_result.append(code)
            try:
                merge_history[word]=tt_result[0]
            except:
                pass
            continue            
        else:
            head_array=[]
            for one in united:
                sp=one.split('#')
                head=sp[0]
                if mapping.get(head,-100)==-100:
                    continue
                merge_history[head]=head
                head_array.append(head)
                collection_entity.delete(ids=[mapping[head]['ID']])
                for u in range(1,len(sp)):
                    if mapping.get(sp[u],-100)==-100 :
                        continue
                    try:
                        local_label=full_word_table_entity[head]['标签']
                        tail_label=full_word_table_entity[sp[u]]['标签']
                        no_sen=temp_save_dict[local_label][head]
                        no_sen=temp_save_dict[tail_label][sp[u]]
                    except:
                        continue
                    temp_save_dict[local_label][head]=[temp_save_dict[local_label][head]].append(temp_save_dict[tail_label][sp[u]])
                    try:
                        del temp_save_dict[tail_label][sp[u]]
                        del full_word_table_entity[sp[u]]
                    except:
                        continue
                    print("删除{},合并到{}".format(sp[u],head))
                    merge_history[sp[u]]=head
                    collection_entity.delete(ids=[mapping[sp[u]]['ID']])
    #三元组替换同义词
    full_triplet_final=[]
    for triplet in full_triplet:
        head=triplet['head']
        tail=triplet['tail']
        relation=triplet['relation']
        if merge_history.get(head,-100)!=-100:
            head=merge_history[head]
        if merge_history.get(tail,-100)!=-100:
            tail=merge_history[tail]
        full_triplet_final.append({
                    'head':head,
                    'tail':tail,
                    'relation':relation,
                    'textID':triplet['textID'],
                    'article':triplet['article']                       
                    })
    return full_word_table_entity,full_triplet_final

if __name__ == '__main__':
    # 获取日期参数
    date = config.args.date  
    global_distance = 0.3
    
    # 定义基础的总输入和总输出路径
    base_origin_path = "./data/{}/step1_result/".format(date)
    base_result_path = "./data/{}/step2_result/".format(date)

    if not os.path.exists(base_origin_path):
        print(f"❌ 错误：找不到基础路径 {os.path.abspath(base_origin_path)}")
        import sys
        sys.exit()

    num = 0
    print("🚀 开始自动扫描目录结构...")
    
    # 核心魔法：os.walk 自动遍历所有深度的文件夹
    for root, dirs, files in os.walk(base_origin_path):
        # 1. 检查当前文件夹下有没有真实的文件（过滤掉隐藏文件）
        valid_files = [f for f in files if not f.startswith('.')]
        
        # 如果当前文件夹是个“空壳”，直接跳过
        if len(valid_files) == 0:
            continue
            
        # 获取相对路径，用于在结果文件夹中等比例复刻目录！
        rel_path = os.path.relpath(root, base_origin_path)
        
        # 【换壳】：提取出最底层文件夹的名字作为具体技术和算法模型名
        tech_model_name = os.path.basename(root)
        
        # 拼接最终的保存路径
        t = os.path.join(base_result_path, rel_path)
        
        num += 1
        print(f"\n=========================================")
        # 【换壳】：探测区 -> 具体技术和算法模型
        print(f"🎯 自动锁定技术/算法任务: 【{rel_path}】")
        
        # 【换壳】
        output_filename = "{}{}最终合并结果.txt".format(date, tech_model_name)
        
        # 3. 断点续传检查
        if os.path.exists(os.path.join(t, output_filename)):
            print(f"⏩ 略过已完成的具体技术和算法模型: {tech_model_name}")
            continue
        else:
            os.makedirs(t, exist_ok=True) # 自动连带父目录一起创建出来
            
        # --- 下面是你原汁原味的处理逻辑 ---
        chroma_client = chromadb.Client()
        collection_entity = chroma_client.create_collection(name='collection_entity{}'.format(str(num)))
        
        # 把这个最小的包含文件的文件夹路径传给 step1_load
        full_word_table_entity, full_triplet, temp_save_dict, figure_dict, file_number, collection_entity = step1_load(root, collection_entity)

        word_table_entity_mirror = eval(str(full_word_table_entity))
        full_word_table_entity, full_triplet_final = step2_merge(word_table_entity_mirror, full_triplet, collection_entity)      

        mirror_temp_save_dict = eval(str(temp_save_dict))

        mirror_temp_save_dict["实例词表"] = full_word_table_entity
        mirror_temp_save_dict["全部三元组"] = full_triplet_final
        mirror_temp_save_dict["数值表"] = figure_dict
        mirror_temp_save_dict["论文数量"] = file_number
        
        # 4. 保存文件
        with open(os.path.join(t, output_filename), "w", encoding='utf-8') as f:
            f.write(str(mirror_temp_save_dict))
            
        print(f"✅ 具体技术和算法模型【{tech_model_name}】合并并保存成功！")