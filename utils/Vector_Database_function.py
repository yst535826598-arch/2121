from openai import OpenAI
import config
config.init_config()
client = OpenAI(
    api_key=config.args.APIKEY,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url=config.args.URL  # 百炼服务的base_url
)
def my_embeddings_fuction(word,client=client):
    completion = client.embeddings.create(
        model=config.args.embedding_model,
        input=word,
        dimensions=512,
        encoding_format="float"
    )
    c=eval(completion.model_dump_json())
    embedding_result=c['data'][0]['embedding']
    return embedding_result
def add_attribute_to_chromadb(mydb,attribute,emb):#添加词
    id="id{}".format(int(mydb.count())+1)
    mydb.add(
        embeddings=[emb],
        documents=[attribute],
        ids=[id]
    )    
    return 1
def determine_attribute(mydb,emb,num=5):
    results = mydb.query(
    query_embeddings=[emb],
    n_results=num)
    attribute_in_db=results["documents"][0][0:num]
    attribute_id_in_db=results["ids"][0][0:num]
    return attribute_in_db,attribute_id_in_db
def determine_attribute_distance(mydb,emb,num=5):#返回最像的一个词,输入向量
    results = mydb.query(
    query_embeddings=[emb],
    n_results=num)
    attribute_in_db=results["documents"][0][0:num]
    attribute_id_in_db=results["ids"][0][0:num]
    attribute_distance_in_db=results["distances"][0][0:num]
    return attribute_in_db,attribute_id_in_db,attribute_distance_in_db