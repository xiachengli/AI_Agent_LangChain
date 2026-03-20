from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

db_user = "root"
db_password = "root"
db_host = "localhost"
db_port = 3306
db_datebase = "test"
# 连接Mysql数据库
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_datebase}")

print("哪种数据库：", db.dialect)
print("数据库表：", db.get_usable_table_names())

# 执行SQL查询
query = "SELECT count(*) FROM user;"
print("查询结果：", db.run(query))


# create_sql_query_chain的使用：将自然语言查询转换为SQL查询
# 加载模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 换小模型
    temperature=0.7,
    num_ctx=1024,  # 上下文长度
    top_p=0.9,
)

# 创建SQL查询链
sql_query_chain = create_sql_query_chain(   
    llm=llm,
    db=db,
    )       
# 执行SQL查询链
result = sql_query_chain.invoke({"question": "user表中有多少条记录。"})
print("SQL查询链结果：", result)





