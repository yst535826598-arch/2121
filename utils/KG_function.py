from py2neo import * 
import config
config.init_config()
graph= Graph(config.args.KGlink, name=config.args.KGname,auth=(config.args.KGcount, config.args.KGcode))
def create_node_plus(type,name,data=False,frequency=False):
    node_matcher = NodeMatcher(graph)
    node = node_matcher.match(type).where(name=name).first()
    if node is None:
        #print("不存在，已创建")
        a = Node(type,name=name)
        if data:
            a.update(data)
        a['frequency']=1
        graph.create(a)
        return a
    else:
        if frequency:#允许修改frequency
            if data:
                node.update(data)
                example=int(node['frequency'])+1
                node['frequency']=example
                graph.push(node)
                node = node_matcher.match(type).where(name=name).first()
                return node
            else:      #print("存在，已返回")
                example=int(node['frequency'])+1
                node['frequency']=example
                graph.push(node)
                node = node_matcher.match(type).where(name=name).first()
                return node
        else:#不允许修盖frequency，等待判断
            if data:
                node.update(data)#更新定义，但不知道这条边是不是新的，不更新frequency
                #example=int(node['frequency'])+1
                #node['frequency']=str(example)
                graph.push(node)
                node = node_matcher.match(type).where(name=name).first()
                return node
            else:      #print("存在，已返回")
                return node
def if_relation_exist_plus(node1,node2,relation,save,data=False,frequency=True):
    if data is not False:
        t_triplet={'head':node1['name'],
                    'tail':node2['name'],
                    'relation':str(relation),
                    'data':str(data)}
    else:
        t_triplet={'head':node1['name'],
                    'tail':node2['name'],
                    'relation':str(relation),
                    'data':'空'}
    save.append(t_triplet)
    relmatcher=RelationshipMatcher(graph) 
    relationship = relmatcher.match((node1,node2), r_type=str(relation)).first()
    #print(relationship)
    if relationship is None:
        #print("不存在此关系，已创建")
        ab_relation = Relationship(node1, str(relation), node2)
        ab_relation['frequency']=1
        if data:
            ab_relation['来源1']=data
        #ab_relation['text1']=text
        #relationship.update(relation['data'])
        #未来这里可能需要把原因作为实例加上
        graph.create(ab_relation)  # 创建节点和关系
        #return True
    else:
        #print("存在此关系,不做处理")
        example=int(relationship['frequency'])+1
        if frequency:
            relationship['frequency']=example
        if data:
            relationship['来源{}'.format(example)]=data
        #relationship.update(relation['data'])
        #未来这里可能需要把原因作为实例加上
        graph.push(relationship)
    return save