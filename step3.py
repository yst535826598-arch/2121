import os
import pickle  # 新增：用于高效、小体积保存数据的原生库
import copy    # 新增：用于替代 eval(str()) 的深拷贝库
from py2neo import *
import chromadb
from utils.KG_function import *
from utils.LLM_function import *
from utils.Vector_Database_function import *
import config

config.init_config()

# 【换壳】：areas_path -> tech_models_path
def step1_load_entitys_embedding(load_entitys_embedding,tech_models_path,final_path,date):
    embedding_load_history={}
    full_word_table_entitys={}
    if load_entitys_embedding:
        print("正在加载实体嵌入表")
        # 【修改】使用 pickle 二进制读取，防内存溢出
        with open(os.path.join(final_path,"{}实体嵌入表加载历史.pkl".format(date)),'rb') as file:
            embedding_load_history=pickle.load(file)
        # 【修改】使用 copy.deepcopy 替代 eval(str())
        full_word_table_entitys=copy.deepcopy(embedding_load_history['full_word_table_entitys'])
        
        print(len(full_word_table_entitys))
        for en in full_word_table_entitys.keys():
            em=full_word_table_entitys[en]['嵌入向量']
            add_attribute_to_chromadb(collection_entitys,en,em)
        
    tech_models=os.listdir(tech_models_path)
    for t_tech_model in tech_models:
        if load_merge_history is True:
            print("载入对齐历史，跳过实例载入")
            break
        tech_model=t_tech_model
        if embedding_load_history.get(tech_model,-100)!=-100:
            print("已恢复并略过{}".format(tech_model))
            continue
        folder_path=os.path.join(tech_models_path,t_tech_model)
        tech_model_files=os.listdir(folder_path)
        for tech_model_file in tech_model_files:
            if tech_model_file.find(str(date))==-1:
                continue
            # 原始输入文件仍然使用 eval 读取，保持与你原数据兼容
            with open(os.path.join(folder_path,tech_model_file),'r',encoding='utf-8') as file:
                texts=file.readline()
                tech_model_dict=eval(texts)
            
            temp_word_table_entity=tech_model_dict["实例词表"]
            temp_triplet=tech_model_dict["全部三元组"]
                    #add_attribute_to_chromadb(collection_addition,word,emb)
            for word in temp_word_table_entity.keys():#full_word_table_entity
                # 💡 核心修复：跳过空实体，防止 API 报错
                if not word or str(word).strip() == "":
                    continue
                if full_word_table_entitys.get(word,-100)==-100 :
                    full_word_table_entitys[word]=temp_word_table_entity[word]
                    emb=temp_word_table_entity[word]['嵌入向量']
                    t_label=temp_word_table_entity[word]['类型']
                    add_attribute_to_chromadb(collection_entitys,word,emb)
                
            for tlabel in tech_model_dict.keys():
                if tlabel=='全部三元组' or tlabel=='实例词表' or tlabel=='扩展词表'or tlabel=="词类型嵌入表"or tlabel=="属性表"or tlabel=="论文数量":
                    continue
                for entitys in tech_model_dict[tlabel].keys():
                    # 💡 核心修复：跳过空实体，防止 API 报错
                    if not entitys or str(entitys).strip() == "":
                        print("⚠️ 警告：检测到空实体名称，已跳过嵌入计算...")
                        continue
                        
                    if full_word_table_entitys.get(entitys,-100)==-100:
                        full_word_table_entitys[entitys]={}
                        full_word_table_entitys[entitys]['标签']=tlabel

                        full_word_table_entitys[entitys]['唯一性']="不明"
                        full_word_table_entitys[entitys]['类型']="不明"
                        emb=my_embeddings_fuction(entitys)
                        full_word_table_entitys[entitys]['嵌入向量']=emb
                        add_attribute_to_chromadb(collection_entitys,entitys,emb)

            
        embedding_load_history[tech_model]=1
        embedding_load_history['full_word_table_entitys']=full_word_table_entitys
        # 【修改】使用 pickle 二进制保存
        with open(os.path.join(final_path,"{}实体嵌入表加载历史.pkl".format(date)), "wb") as f:
            pickle.dump(embedding_load_history, f)
    return full_word_table_entitys

def step2_merge_all(load_merge_history,recover,full_word_table_entitys,final_path,date):
    if load_merge_history is False:
        merage_protect_entity={}
        merge_history_entitys={}
        save_the_graph={}
        # 【修改】深拷贝替代 eval(str())
        word_table_entity_mirror=copy.deepcopy(full_word_table_entitys)
        full=len(word_table_entity_mirror)
        if recover is True:
            # 【修改】使用 pickle 二进制读取
            with open(os.path.join(final_path,"{}对齐中间备份.pkl".format(date)),'rb') as file:
                save_the_graph=pickle.load(file)
            merge_history_entitys=save_the_graph['实例对齐记录']
            full_word_table_entitys=save_the_graph['对齐后实例词表']
            flag=0
            for word in word_table_entity_mirror.keys():#恢复对齐历史
                flag=flag+1
                res=round((flag/full),3)
                print("恢复对齐进度{}%".format(res*100))
                if merge_history_entitys.get(word,-100)!=-100:
                    emb=word_table_entity_mirror[word]['嵌入向量']
                    similar,similarID,similar_distance=determine_attribute_distance(collection_entitys,emb,2)
                    if similar[0]==word:
                        collection_entitys.delete(ids=[similarID[0]])
                    continue
        flag=0
        for word in word_table_entity_mirror.keys():#以后加个已完成百分比
            flag=flag+1
            res=round((flag/full),3)
            print("对齐进度{}%".format(res*100))
            if merge_history_entitys.get(word,-100)!=-100:
                continue
            # 【修改】深拷贝替代 eval(str())
            merge_history_entitys[word]=copy.deepcopy(full_word_table_entitys[word])
            merge_history_entitys[word]['名称']=word
            emb=word_table_entity_mirror[word]['嵌入向量']
            similar,similarID,similar_distance=determine_attribute_distance(collection_entitys,emb,10)
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
                                            
            # ==============================================================
            # 💡 【超级提速拦截器】：如果没有找到相似词（列表里只有自己）
            # 绝对不要去调用大模型！直接本地确认，秒速跳过！
            # ==============================================================
            if len(array_4_aligen) <= 1:
                merage_protect_entity[word] = 1
                if word in mapping:
                    collection_entitys.delete(ids=[mapping[word]['ID']])
                
                # 触发保存机制验证
                if flag%10==0 and flag!=0:
                    print("中间过程备份")
                    save_the_graph['对齐后实例词表']=full_word_table_entitys
                    save_the_graph['实例对齐记录']=merge_history_entitys
                    with open(os.path.join(final_path,"{}对齐中间备份.pkl".format(date)), "wb") as f:
                        pickle.dump(save_the_graph, f)
                continue
            # ==============================================================

            united=level2_merge_special(array_4_aligen)
            if str(united).find('NO')!=-1 or str(united).find('ERROR')!=-1:
                tt_result=[]
                for code in array_4_aligen:
                    if code==word:
                        continue
                    else:
                        tt_result.append(code)
                try:
                    merge_history_entitys[word]=copy.deepcopy(full_word_table_entitys[tt_result[0]])
                    merge_history_entitys[word]['名称']=tt_result[0]
                    merage_protect_entity[word]=1
                    collection_entitys.delete(ids=[mapping[word]['ID']])
                except:
                    print("array为空")
                continue     
            else:
                print(united)
                head_array=[]
                for one in united:
                    sp=one.split('#')
                    head=sp[0]
                    if mapping.get(head,-100)==-100:
                        print("合并时出现未见词，跳过")
                        continue
                    merge_history_entitys[head]=copy.deepcopy(full_word_table_entitys[head])
                    merge_history_entitys[head]['名称']=head
                    head_array.append(head)
                    merage_protect_entity[head]=1
                    collection_entitys.delete(ids=[mapping[head]['ID']])
                    for u in range(1,len(sp)):
                        if mapping.get(sp[u],-100)==-100 :
                            print("合并时出现未见词，跳过")
                            continue
                        try:
                            if sp[u]!=head and merage_protect_entity.get(sp[u],-100)==-100:
                                del full_word_table_entitys[sp[u]]
                        except:
                            print("{}已经被删除过".format(sp[u]))
                            continue
                        print("删除{},合并到{}".format(sp[u],head))
                        try:
                            merge_history_entitys[sp[u]]=copy.deepcopy(full_word_table_entitys[head])
                        except:
                            merge_history_entitys[sp[u]]=copy.deepcopy(merge_history_entitys[head])
                        merge_history_entitys[sp[u]]['名称']=head
                        collection_entitys.delete(ids=[mapping[sp[u]]['ID']])
            if flag%10==0 and flag!=0:#每10个保存一次
                print("中间过程备份")
                save_the_graph['对齐后实例词表']=full_word_table_entitys
                save_the_graph['实例对齐记录']=merge_history_entitys
                with open(os.path.join(final_path,"{}对齐中间备份.pkl".format(date)), "wb") as f:
                    pickle.dump(save_the_graph, f)
        save_the_graph['对齐后实例词表']=full_word_table_entitys
        save_the_graph['实例对齐记录']=merge_history_entitys
        with open(os.path.join(final_path,"{}对齐记录.pkl".format(date)), "wb") as f:
            pickle.dump(save_the_graph, f)
    if load_merge_history is True:
        with open(os.path.join(final_path,"{}对齐记录.pkl".format(date)),'rb') as file:
            save_the_graph=pickle.load(file)
        merge_history_entitys=save_the_graph['实例对齐记录']
        full_word_table_entitys=save_the_graph['对齐后实例词表']
    return merge_history_entitys,full_word_table_entitys

def step3_load2neo4j(tech_model_name,tech_model_dict,graph_label_entitys,graph_label_attribute,final_entitys,final_tri):
    paper_number=int(tech_model_dict.get('论文数量', 1))
    
    node_properties = {'论文数量': paper_number}
    attribute_dict = tech_model_dict.get('数值表', {})
    
    # 💡 核心改造 1：拦截那 5 个原本是属性，现在要提拔为实体的特定词！
    special_info_labels = ['工作创建时间', '工作处理阶段说明', '工作完成时间', '工作完成人姓名', '工作内容所属单位']
    extracted_special_infos = [] # 用于暂存这些拦截下来的词
    
    for attr_label in attribute_dict.keys():
        attr_values = []
        for item in attribute_dict[attr_label]:
            t = item['内容'].split('#')
            try:
                entity_data = t[1] # 具体的数值或名字
                message = t[0]     # 大模型总结的含义
            except:
                continue
            
            # 如果是那 5 个特殊信息，拦截下来，不进入 node_properties！
            if attr_label in special_info_labels:
                extracted_special_infos.append({
                    'label': attr_label,
                    'name': entity_data, 
                    'description': message
                })
            else:
                if attr_label == '其他数值或符号':
                    attr_values.append(entity_data)
                else:
                    attr_values.append(f"{message}:{entity_data}")
                
        if len(attr_values) > 0 and attr_label not in special_info_labels:
            unique_values = list(set(attr_values))
            node_properties[attr_label] = " ; ".join(unique_values)

    # ==== 步骤 1：创建 一级节点（中心算法/模型）====
    tech_model_node=create_node_plus(graph_label_entitys,tech_model_name,data=node_properties)
    tech_model_node.add_label('具体技术和算法模型')
    graph.push(tech_model_node)
    
    final_entitys[tech_model_name]={'type':graph_label_entitys, 'name':tech_model_name, 'additional_type':'具体技术和算法模型', 'data':'空'}
    
    # ==== 步骤 2：动态建立 二级大类路由节点池 ====
    # 这是一个辅助函数：如果节点不存在就创建它，并且跟一级中心节点连上
    level2_nodes = {}
    def get_level2_node(macro_category, current_tri):
        if macro_category not in level2_nodes:
            l2_node = create_node_plus(graph_label_entitys, macro_category)
            l2_node.add_label(macro_category)
            graph.push(l2_node)
            # 💡 一级节点 -> [包含大类] -> 二级节点
            current_tri = if_relation_exist_plus(tech_model_node, l2_node, "包含大类", current_tri, data=tech_model_name)
            level2_nodes[macro_category] = l2_node
        return level2_nodes[macro_category], current_tri

    # ==== 步骤 3：处理拦截下来的 5 个特殊工作信息（提拔为 三级节点）====
    for info in extracted_special_infos:
        l_label = info['label']  # 比如 '工作创建时间'
        l_name = info['name']    # 比如 '2023年'
        # 从配置中找它的二级大类（默认是 '工作完成信息'）
        macro_category = config.args.KGmacro_mapping.get(l_label, '工作完成信息')
        l2_node, final_tri = get_level2_node(macro_category, final_tri)
        
        # 创建三级实体
        a_node = create_node_plus(graph_label_entitys, l_name, data={'描述含义': info['description']})
        a_node.add_label(l_label)
        graph.push(a_node)
        # 💡 二级节点 -> [具体分类标签] -> 三级节点
        final_tri = if_relation_exist_plus(l2_node, a_node, l_label, final_tri, data=tech_model_name)

    # ==== 步骤 4：处理常规抽取的实例实体（创建 三级节点）====
    for label in tech_model_dict.keys():
        number_for_label=0
        if label in ['全部三元组', '实例词表', '扩展词表', '词类型嵌入表', '数值表', '论文数量']:
            continue
            
        # 💡 根据 config.py 寻找这个 label 对应的宏观大类（二级节点名）
        macro_category = config.args.KGmacro_mapping.get(label, '其他')
        if not macro_category: macro_category = '其他'
        l2_node, final_tri = get_level2_node(macro_category, final_tri)

        for entitys in tech_model_dict[label].keys():
            if merge_history_entitys.get(entitys,-100)!=-100:
                entity_detial=merge_history_entitys[entitys]
                entitys=entity_detial['名称']
            else:
                try:
                    entity_detial=full_word_table_entitys[entitys]
                except:
                    continue
            
            # 创建三级实体
            a_node=create_node_plus(graph_label_entitys,entitys)
            final_entitys[entitys]={'type':[graph_label_entitys], 'name':entitys, 'additional_type':'空', 'data':'空'}
            
            # 💡 核心转变：实体不再直连中心一级节点，而是连到了所属的二级大类节点！
            final_tri=if_relation_exist_plus(l2_node, a_node, label, final_tri, data=tech_model_name)
            a_node.add_label(label)
            graph.push(a_node)
            number_for_label += 1
            
        # （可选保留）一级中心节点记录各细分标签的统计量
        tech_model_node=create_node_plus(graph_label_entitys,tech_model_name,data={'{}标签下节点数量'.format(str(label)):number_for_label})
    
    # 步骤 5：处理原文三元组逻辑（连接三级节点之间的相互关系，维持原状）
    tri_array=tech_model_dict.get('全部三元组', [])
    for tri in tri_array:
        head=tri['head']
        if merge_history_entitys.get(head,-100)!=-100:
            head=merge_history_entitys[head]['名称']
        else:
            continue
        tail=tri['tail']
        if merge_history_entitys.get(tail,-100)!=-100:
            tail=merge_history_entitys[tail]['名称']
        else:
            continue
        relation=tri['relation']
        textID=tri['textID']
        article=tri['article']
        data=article+'#'+textID
        head_node=create_node_plus(graph_label_entitys,head)
        tail_node=create_node_plus(graph_label_entitys,tail)
        final_tri=if_relation_exist_plus(head_node,tail_node,relation,final_tri,data=data)
        
    return final_entitys,final_tri


if __name__ == '__main__':
    date=config.args.date
    global_distance=0.3
    #一般开始时候全是False，怕中间出错。全跑完以后系统中出现缓存文件后，第二个和第四个改成True
    load_entitys_embedding=False#假如step1_load_entitys_embedding以后步骤出问题，改True恢复
    load_merge_history=True#假如step2_merge_all以后步骤出问题或者用step1-2结果重新导入图谱，改True恢复。有完整合并结果才能恢复
    merge_recover=False# 假如step2_merge_all自己出问题，改True恢复对齐进度，load_merge_history=True时不执行。 step2_merge_all非常耗时
    load2neo4j=True#对齐完是否直接导入图谱
    graph_label_entitys='{}图谱实体'.format(date)
    graph_label_attribute='{}图谱数值'.format(date)
    tech_models_path="./data/{}/step2_result/".format(date)
    final_path="./data/{}/step3_result/".format(date)
    os.makedirs(final_path, exist_ok=True)
    
    print(">>> 检查点 1: 准备初始化 ChromaDB...")
    chroma_client = chromadb.Client()
    print(">>> 检查点 2: ChromaDB 客户端初始化完成，准备创建 collection...")
    collection_entitys = chroma_client.create_collection(name='collection_entitys')
    print(">>> 检查点 3: Collection 创建完成，进入 step1...")

    save_the_graph={}
    final_entitys={}
    final_tri=[]
    full_word_table_entitys=step1_load_entitys_embedding(load_entitys_embedding,tech_models_path,final_path,date)
    full_word={}
    full_word['对齐前实例词表']=full_word_table_entitys
    
    with open(os.path.join(final_path,"{}对齐前记录.pkl".format(date)), "wb") as f:
        pickle.dump(full_word, f)
        
    merge_history_entitys,full_word_table_entitys=step2_merge_all(load_merge_history,merge_recover,full_word_table_entitys,final_path,date)
    
    if load2neo4j is False:
        print("略过导入neo4j")
    else:
        tech_model_number=0
        tech_models=os.listdir(tech_models_path)
        for tech_model in tech_models:
            tech_model_number=tech_model_number+1
            print(tech_model+"第{}个具体技术和算法模型，总计{}个".format(tech_model_number,str(len(tech_models))))
            folder_path=os.path.join(tech_models_path,tech_model)
            tech_model_files=os.listdir(folder_path)
            for tech_model_file in tech_model_files:
                with open(os.path.join(folder_path,tech_model_file),'r',encoding='utf-8') as file:
                    texts=file.readline()
                    tech_model_dict=eval(texts)
                final_entitys,final_tri=step3_load2neo4j(tech_model,tech_model_dict,graph_label_entitys,graph_label_attribute,final_entitys,final_tri)        
        
        save_the_graph['对齐后实例词表']=full_word_table_entitys
        save_the_graph['最终实例节点']=final_entitys
        save_the_graph['最终三元组']=final_tri
        
        with open(os.path.join(final_path,"{}图谱备份.pkl".format(date)), "wb") as f:
            pickle.dump(save_the_graph, f)