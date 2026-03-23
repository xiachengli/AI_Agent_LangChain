
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


text = """
   缺少导入语句：代码中直接使用了 JSONLoader 但没有导入它。我添加了正确的导入：from langchain_community.document_loaders import JSONLoader

虚拟环境未激活：langchain_community 包安装在虚拟环境中，需要先激活环境。

缺少依赖包：JSONLoader 需要 jq 包，我帮您安装了它。

JSON 文件格式错误：原文件不是有效的 JSON 数组，我修复了格式并更正了中文字符。

参数设置：JSONLoader 需要 jq_schema 参数，对于数组使用 .[]，并设置 text_content=False 以处理字典内容。 
 """

#递归字符拆分器 常用 保留上下文 智能分段 灵活适配
splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size=50,#每个分块的字符数，默认4000   
    chunk_overlap=5,#分块之间的重叠字符数，确保上下文连续性
    length_function=len,#用于计算分块长度的函数，默认使用 len 函数
    separator="",#分隔符，默认使用换行符,空字符串表示不使用（分隔符优先 ）
    keep_separator=True,#是否保留分隔符，默认保留keep_separator=True 表示在分块中保留分隔符，False 则不保留
)


splitter = CharacterTextSplitter(
    chunk_size=50,#每个分块的字符数
    chunk_overlap=5,#分块之间的重叠字符数，确保上下文连续性
    length_function=len,#用于计算分块长度的函数，默认使用 len 函数
    separator="",#分隔符，默认使用换行符,空字符串表示不使用（分隔符优先 ）
    keep_separator=True,#是否保留分隔符，默认保留keep_separator=True 表示在分块中保留分隔符，False 则不保留
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} size: {len(chunk)}")
    print(f"Chunk {i+1}:\n{chunk}\n")   

