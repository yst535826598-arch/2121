import argparse
from typing import Optional

# 全局配置对象（在导入时初始化为 None）
args: Optional[argparse.Namespace] = None

def init_config():
    global args
    if args is not None:
        return  # 已初始化
    
    figure_array=[
                '地质体埋深范围',   '地质体形态特征',
                '地质体空间尺度',   '地质体形成时代',
                '地质体成因类型',   '地质体接触关系特征',
                '地质体构造属性',   '地质体密度特征',
                '地质体电性特征',   '地质体磁性特征',
                '数据名称',         '数据摘要',
                '数据编号',         '数据空间分辨率',
                '数据采样间隔',      '数据物性响应特征',
                '数据物理量类型',    '数据维度结构',
                '数据表达形式',      '数据噪声水平描述',
                '数据质量等级',      
                '算法参数敏感性',    '算法参数数量',     
                '算法计算复杂度',    '算法计算效率特征',
                '算法基本思想',      '算法类别',
                '算法稳定性特征',    '算法使用场景'
                ]
    
    # 将那5个工作信息“提拔”到实体大类字典中
    entitys_labels_dict={
                '地层':1,                '地球深部构造':1,
                '大地构造单元':1,                '岩浆系统':1,
                '岩石':1,
                '井中地球物理探测仪器':1,         '地热与温度测量仪器':1,
                '地震探测仪器':1,                '实验室与数据处理仪器':1,
                '测井与深部科学钻探装备':1,       '海洋与航空物探平台':1,
                '电法探测仪器':1,                '电磁法探测仪器':1,
                '磁法探测仪器':1,                '跨孔探测仪器':1,
                '辅助与基础设施':1,              '重力探测仪器':1,
                '原始观测数据':1,                '地球物理异常':1,
                '物理模型解释成果':1,            '遥感物理解译标志':1,
                '地热勘察':1,                   '地震采集':1,
                '放射性勘探':1,                  '测井作业':1,
                '电法和电磁采集':1,              '磁法测量':1,
                '综合勘探':1,                   '重力测量':1,
                '信号处理与预处理':1,            '反演算法':1,
                '属性分析与提取':1,             '岩石物理建模':1,
                '成像与偏移':1,                '智能解译与机器学习':1,
                '正演模拟':1,
                '工作创建时间':1,               '工作处理阶段说明':1,
                '工作完成时间':1,               '工作完成人姓名':1,
                '工作内容所属单位':1
                }
    
    macro_mapping_dict = {
        # 1. 地质体
        '地层': '地质体', '地球深部构造': '地质体', '大地构造单元': '地质体', 
        '岩浆系统': '地质体', '岩石': '地质体', '地球物理异常': '地质体',
        
        # 2. 数据
        '原始观测数据': '数据', '物理模型解释成果': '数据', '遥感物理解译标志': '数据',
        
        # 3. 算法 (广义方法与技术：包含处理方法、算法模型、探测仪器硬件 与 勘探采集作业)
        '井中地球物理探测仪器': '算法', '地热与温度测量仪器': '算法', '地震探测仪器': '算法',
        '实验室与数据处理仪器': '算法', '测井与深部科学钻探装备': '算法', '海洋与航空物探平台': '算法',
        '电法探测仪器': '算法', '电磁法探测仪器': '算法', '磁法探测仪器': '算法',
        '跨孔探测仪器': '算法', '辅助与基础设施': '算法', '重力探测仪器': '算法',
        '信号处理与预处理': '算法', '反演算法': '算法', '属性分析与提取': '算法',
        '岩石物理建模': '算法', '成像与偏移': '算法', '智能解译与机器学习': '算法', '正演模拟': '算法',
        '地热勘察': '算法', '地震采集': '算法', '放射性勘探': '算法',
        '测井作业': '算法', '电法和电磁采集': '算法', '磁法测量': '算法',
        '综合勘探': '算法', '重力测量': '算法',
        
        # 4. 工作完成信息 (仅包含纯粹的文本描述及元数据)
        '工作创建时间': '工作完成信息', '工作处理阶段说明': '工作完成信息',
        '工作完成时间': '工作完成信息', '工作完成人姓名': '工作完成信息', '工作内容所属单位': '工作完成信息'
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default="0312")
    parser.add_argument('--APIKEY', type=str, default="sk-7c36adfd428542d0b2400963c1da396c")
    parser.add_argument('--URL', type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")    
    parser.add_argument('--model', type=str, default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument('--embedding_model', type=str, default="text-embedding-v4")#默认是上面URL提供的
    parser.add_argument('--KGlink', type=str, default="bolt://localhost:7687")
    parser.add_argument('--KGname', type=str, default="neo4j")
    parser.add_argument('--KGcount', type=str, default="neo4j")
    parser.add_argument('--KGcode', type=str, default="neo4j@openspg")
    parser.add_argument('--KGentity_labels', type=dict, default=entitys_labels_dict)
    parser.add_argument('--KGfigure_labels', type=dict, default=figure_array)
    parser.add_argument('--KGmacro_mapping', type=dict, default=macro_mapping_dict) # 新增映射参数暴露
    args = parser.parse_args()

    # 可选：打印配置
    print(f"[CONFIG] APIKEY: {args.APIKEY}, URL: {args.URL}, MODEL: {args.model}")