from langchain_community.utilities import SQLDatabase

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
query = "SELECT * FROM user LIMIT 5;"
print("查询结果：", db.run(query))

