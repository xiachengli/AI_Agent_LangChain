

#从txt文件加载文档，向量化后存入chroma数据库
from langchain_community.document_loaders import TextLoader   


#加载文本文件
loader = TextLoader("loader数据/01txt.txt", encoding='utf-8')

#加载文档
docs = loader.load()
print(docs)

#切分
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(
    chunk_size=50,#每个分块的字符数
    chunk_overlap=5,#分块之间的重叠字符数，确保上下文连续性
    length_function=len,#用于计算分块长度的函数，默认使用 len 函数
    separator="",#分隔符，默认使用换行符,空字符串表示不使用（分隔符优先 ）
    keep_separator=True,#是否保留分隔符，默认保留keep_separator=True 表示在分块中保留分隔符，False 则不保留
)

#切分文档
chunks = splitter.split_documents(docs)
print(f"切分后的文档块数量: {len(chunks)}")

#向量化 并存入Chroma数据库
from langchain_ollama import OllamaEmbeddings

# 创建嵌入模型
embeddings = OllamaEmbeddings(model="qwen3-embedding")

# 创建Chroma数据库实例
from langchain_community.vectorstores import Chroma
try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="my_collection",
        persist_directory="./chroma_db"
    )
    print("Chroma数据库创建成功")
except Exception as e:
    print(f"创建数据库时出错: {e}")       





