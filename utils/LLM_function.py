from openai import OpenAI
import time
import config
config.init_config()

# 基础调用函数不变
def ask_llm_base(question,system_prompt='你是一名深部地球物理探测与计算机算法专家，我需要你回答一些专业问题。',mykey=config.args.APIKEY,mybase_url=config.args.URL,mymodel=config.args.model):
    time.sleep(0.05)
    client = OpenAI(
        api_key=mykey,  
        base_url=mybase_url,  
    )
    chat_completion = client.chat.completions.create(
    temperature=0.1,
    model=mymodel,
    messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
        ],
    max_tokens=1500, # 加大 token，以适应批处理输出
    )
    return chat_completion.choices[0].message.content

def llm_check_part_array(p1,p2,time=3,mymodel='default'):
    flag=time
    while(flag):
        if mymodel!='default':
            answer=ask_llm_base(question=p2,system_prompt=p1,mymodel=mymodel)
        else:
            answer=ask_llm_base(question=p2,system_prompt=p1)
        print(answer)
        start=answer.find('ARRAYSTART')
        end=answer.find('ARRAYEND')
        if start!=-1 and end!=-1:
            mydict=answer[start+11:end]
            try:
                start=mydict.find('[')
                end=mydict.rfind(']')
                mydict=mydict[start:end+1]
                mydict=mydict.replace('/','')
                mydict=eval(mydict)
                if type(mydict) is not list or type(mydict) is tuple:
                    raise KeyError
                break
            except:
                print("回答不合格，重复中")
                flag=flag-1
                continue
        else:            
            print("回答不合格，重复中")
            flag=flag-1
            continue
    if flag==0:
        return "ERROR"
    return mydict 

def llm_check_part_dict(p1,p2,time=3,mymodel='default'):
    flag=time
    while(flag):
        if mymodel!='default':
            answer=ask_llm_base(question=p2,system_prompt=p1,mymodel=mymodel)
        else:
            answer=ask_llm_base(question=p2,system_prompt=p1)
        print(answer)
        start=answer.find('ARRAYSTART')
        end=answer.find('ARRAYEND')
        if start!=-1 and end!=-1:
            mydict=answer[start+11:end]
            try:
                start=mydict.find('{')
                end=mydict.rfind('}')
                mydict=mydict[start:end+1]
                mydict=mydict.replace('/','')
                mydict=eval(mydict)
                if type(mydict) is not dict or type(mydict) is tuple:
                    print(mydict)
                    raise ValueError("发现非字典输出")
                break
            except:
                print("回答不合格，重复中")
                flag=flag-1
                continue
        else:            
            print("回答不合格，重复中")
            flag=flag-1
            continue
    if flag==0:
        return "ERROR"
    return mydict 

def level2_check(question,answer):
    return True, "跳过"

# ==== 【深度重构：强迫大模型抽取细粒度实例，同时加入字数与词性限制】 ====

def level1_entity(text):
    p1="你是一名深部地球物理探测与计算机算法专家。你的核心任务是从文本中提取【极其具体、落脚到实物/具体代码实现级别】的实例名称。英文请翻译为中文。"
    p2="我会给出一段话，请你提取其中具体的硬件工具、具体的算法变体名称。\n"
    p2+="【极度警告1】：绝对不可提取软件的设计理念、架构特征或物理场的几何属性！必须是确切存在的工具或方法名！\n"
    p2+="【极度警告2】：提取的名称必须是精简的名词短语（15字内）。\n"
    p2+="【极度警告3】：严禁抽取单个字母（如'E', 'O'）、纯数学公式/符号（如'lₖ,ₙˡ(ω)'）或过于宽泛的孤立名词（如'卫星', '算法'）！\n"
    p2+="【反面教材（坚决不要抽）】：'插件化架构设计'（这是设计理念）、'磁层环电流几何结构'、'反演算法'、'E'、'lₖ,ₙˡ(ω)'、'卫星'。\n"
    p2+="【正面教材（必须这样抽）】：'非线性共轭梯度算法'、'宽频带海底地震仪(OBS)'、'ResNet-50深度残差网络'、'MagTFs工具箱'、'GOCE重力卫星'。\n"
    p2+="回答格式必须严格遵守python字符串数组的格式：ARRAYSTART ['细粒度实例1', '细粒度实例2'] ARRAYEND。你要抽取的一段话是："+text
    my_time=3
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=2)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"

def level1_entity_forPhenomenon(text):
    p1="你是一名深部地球物理探测与计算机算法专家。你的任务是从文本中抽取【极其具体、特异性强】的地质、地球物理或计算现象实例。英文名词请翻译为中文。"
    p2="我会给出一段话，请提取具体的现象实例。必须是精简的具体现象【名词短语】（严格控制在5-15个字以内）！\n"
    p2+="【极度警告1】：严禁抽取宽泛的字典名词或高度抽象的现象大类！也不可抽取物性的几何结构。\n"
    p2+="【极度警告2】：绝对不可把带有动词的长句子、事件过程当作实体抽出！必须剥离冗长的背景描述！\n"
    p2+="【极度警告3】：严禁抽取单个字母、数学符号或极度宽泛的孤立词汇！\n"
    p2+="【反面教材（太长或太抽象）】：'重力高异常'、'晚海西期地幔柱上涌...'、'磁层环电流几何结构'、'O'。\n"
    p2+="【正面教材（精准简练）】：'塔里木中央局部重力高'、'塔北缘壳幔部分熔融'、'双向剪切型物质交换'\n"
    p2+="回答格式要求：ARRAYSTART ['细粒度现象1', '细粒度现象2'] ARRAYEND。你要抽取的一段话是："+text
    my_time=3
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=2)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"

def level1_entity_forConcept(text):
    p1="你是一名深部地球物理探测与计算机算法专家。你的任务是从文本中抽取文献真正使用或提出的【最底层的具体算法工具、特定神经网络模型名称】。英文请翻译为中文。"
    p2="我会给出一段话，请提取具体的方法模型或工具实体。\n"
    p2+="【极度警告1】：大类本体名词、软件的设计模式、架构理念对知识图谱毫无意义！严禁抽取泛泛而谈的分类词或设计思路！\n"
    p2+="【极度警告2】：名称必须是精练的名词词组（15字内），坚决剥离研究背景修饰语。\n"
    p2+="【极度警告3】：严禁抽取数学公式、纯数字、单个字母变量（如'E', 'O', 'lₖ,ₙˡ(ω)'）！\n"
    p2+="【反面教材（坚决不要抽）】：'大地电磁测深'、'插件化架构设计'、'基于组件的软件开发'、'lₖ,ₙˡ(ω)'。\n"
    p2+="【正面教材（精准简练）】：'三维大地电磁正演算法'、'自适应加权高斯-牛顿反演'、'U-Net智能断层解译网络'、'MagTFs工具箱'。\n"
    p2+="回答格式要求：ARRAYSTART ['细粒度核心算法1', '细粒度具体工具2'] ARRAYEND。你要抽取的一段话是："+text
    my_time=3
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=2)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"

def level1_entity_forExample(text):
    p1="你是一名深部地球物理探测与计算机算法专家。你的任务是从文本中抽取绝对具象的【真实测试数据集、标准测试模型或真实空间工区名称】。英文请翻译为中文。"
    p2="我会给出一段话，请提取不可再分的具体测试案例实例，且必须是【15字以内的名词词组】。\n"
    p2+="【反面教材】：'基准测试模型'、'塔里木盆地北缘在古生代形成的复杂断裂带工区'、'卫星'\n"
    p2+="【正面教材】：'Marmousi二维基准模型'、'SEG/EAGE盐丘模型'、'塔北隆起工区'、'GRACE重力卫星'\n"
    p2+="回答格式要求：ARRAYSTART ['具体测试实例1'] ARRAYEND。你要抽取的一段话是："+text
    my_time=3
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=2)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"

def level1_entity_forfigure(text):
    p1="你是一名深部地球物理探测与计算机算法专家，你的任务是从一段话中抽取具备实际意义的数字或者符号形式的物理参数或算法复杂度。注意：由于原文是英文文献，请你务必将数值含义翻译为标准中文专业术语后输出。你应全程使用中文回答。"
    p2="我会给出一段话。这些信息大多表现为带有单位的数值或符号，例如：计算耗时20s中的20s、空间分辨率10m中的10m、深度35km中的35km等。你需要注意你应同时用一个词来回答这个数值的含义。如果文字中没有任何数值或者符号类信息请回答无。回答格式样例:ARRAYSTART ['含义1#数字或者符号1','含义2#数字或者符号2'] ARRAYEND  没有任何符合要求信息时回答ARRAYSTART ['无#无'] ARRAYEND 你要抽取的一段话是:"+text
    my_time=2
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=2)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"

def level1_entity_multiple_strategy(text,strategy='single',repetitions=1):
    flag=0
    result_dict={}
    result_array=[]
    result_dict_figure={}
    result_array_figure=[]
    if strategy=='single':
        while(flag<repetitions):
            flag=flag+1
            entitys=level1_entity(text)
            for i in entitys:
                if result_dict.get(i,-100)==-100:
                    result_dict[i]=1
    if strategy=='multiple':
        while(flag<repetitions):
            flag=flag+1
            entitys_Phenomenon=level1_entity_forPhenomenon(text)
            entitys_Example=level1_entity_forExample(text)
            entitys_Concept=level1_entity_forConcept(text)
            entitys_figure=level1_entity_forfigure(text)
            for i in entitys_Concept:
                if result_dict.get(i,-100)==-100:
                    result_dict[i]=1
            for i in entitys_Example:
                if result_dict.get(i,-100)==-100:
                    result_dict[i]=1
            for i in entitys_Phenomenon:
                if result_dict.get(i,-100)==-100:
                    result_dict[i]=1
            for i in entitys_figure:
                if result_dict_figure.get(i,-100)==-100 and i!='无#无':
                    result_dict_figure[i]=1
    for entity in result_dict.keys():
        result_array.append(entity)
    for entity in result_dict_figure.keys():
        result_array_figure.append(entity)
    return result_array,result_array_figure


def level1_entity_label_batch(entity_list, text, labels_list):
    if not entity_list or len(entity_list) == 0:
        return {}
        
    p1 = "你是一名深部地球物理探测与计算机算法分类专家。我为你提取出了一批【非常具体的底层实体实例】，你的任务是为它们打上合适的【高层抽象分类标签】。"
    p2 = "这段话是：\\n" + text + "\\n\\n"
    p2 += "从这段话中，我提取了以下极其具体的细粒度实例列表：\\n" + str(entity_list) + "\\n\\n"
    p2 += "你【只能】从以下抽象的大类本体标签范围中进行选择：\\n" + str(labels_list) + "\\n\\n"
    p2 += "任务要求：\\n1. 请根据上下文，为每一个具体实例，从给定的抽象标签库中选择【最合适的一个大类】。\\n"
    p2 += "例如：如果你看到实例是'Marmousi二维测试模型'，你应该为它选择标签'物理模型解释成果'。如果实例是'自适应高斯牛顿反演'，应选择'反演算法'。\\n"
    p2 += "2. 绝对不能生造标签范围以外的类别！如果确实不知如何分类，可回答'其他'，但尽量归类。\\n"
    p2 += "3. 严格使用Python字典格式回答：键是细粒度实例原词，值是你选的抽象大类。\\n"
    p2 += "回答格式样例: ARRAYSTART {'ResNet-50深度残差网络':'智能解译与机器学习', '塔里木盆地奥陶系碳酸盐岩':'地层与岩性'} ARRAYEND"
    
    my_time = 3
    while(my_time > 0):
        mydict = llm_check_part_dict(p1, p2, time=2)
        if mydict == "ERROR":
            my_time -= 1
            continue
            
        final_dict = {}
        for entity, label in mydict.items():
            # 模糊匹配标签纠错
            matched = False
            for std_label in labels_list:
                if label == std_label or label.find(std_label) != -1 or std_label.find(label) != -1:
                    final_dict[entity] = std_label
                    matched = True
                    break
            
            if not matched:
                final_dict[entity] = label
                
        return final_dict
        
    return "ERROR"

# ==== 三元组抽取 ====
def level2_relation_extract(entity,text):
    p1="你是一名深部地球物理探测与计算机算法专家，你的任务是抽取极其具体的细粒度词汇间的直接逻辑关系，并以三元组回答。"
    p2="我从一段文字中提取到了以下底层具体概念:{}。请找到它们间的关系并回答。关系类型必须属于：应用关系（如：某具体算法→某数据）、优化关系（如：某特异性方法→某属性）、因果/组成关系（如：某具体地层→某具体异常现象）、属性关系。回答格式样例:ARRAYSTART ['具体头概念#具体尾概念#关系'] ARRAYEND 这段文字是:".format(str(entity))+text
    my_time=2
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=3)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"

def level2_merge_special(entitys):
    p1="你是一名深部地球物理探测与计算机算法专家，你的任务是判断一批具体词汇是否指代【同一个实体/事物】，并进行同义词合并。"
    p2="候选概念列表:{}。\n".format(str(entitys))
    p2+="请找出指代同一事物的变体词，并按以下【强制规则】进行合并：\n"
    p2+="1. 中英文全称与缩写必须合并（如 'CNN'与'卷积神经网络'，'FWI'与'全波形反演'）。\n"
    p2+="2. 核心实体词 与 其带有后缀（如'算法','模型','工具箱','系统','网络'）的版本必须合并！例如：'MagTFs'与'MagTFs工具箱'，'U-Net'与'U-Net网络模型'必须视为完全相同并合并。\n"
    p2+="3. 合并时，必须选择信息最完整、带有确切后缀的词作为【最优名称】（例如将'MagTFs'合并到'MagTFs工具箱'，选取'MagTFs工具箱'作为最优名称）。\n"
    p2+="4. 【绝对红线】：严禁将不同的变体分支合并（如同态滤波与中值滤波不可合并），严禁将特定子类合并到宽泛父类（如高斯牛顿反演 不可合并到 反演算法）。\n"
    p2+="回答格式样例:ARRAYSTART ['最优名称#被合并变体1#被合并变体2'] ARRAYEND  无词合并回答:ARRAYSTART ['NO'] ARRAYEND "
    my_time=2
    while(my_time>0):
        mydict=llm_check_part_array(p1,p2,time=3)
        if mydict=="ERROR":
            my_time=my_time-1
            continue
        return mydict 
    return "ERROR"