from langchain_community.document_loaders import JSONLoader

#不同数据源使用不用文档加载器，返回List[Document]格式的数据
#JSONLoader 加载JSON文件
#CSVLoader 加载CSV文件
#PyPDFLoader 加载PDF文件 
#文件目录DirectoryLoader 加载目录下的所有文件


loader = JSONLoader("loader数据/01json.json", jq_schema=".[]", text_content=False)
docs = loader.load()
print(docs)