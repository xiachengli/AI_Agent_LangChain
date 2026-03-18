#LangChain基础 含LLM调用和 Chain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#1 加载本地ollama模型,这里使用的是qwen:0.5b,temperature=0.7让输出更灵活
llm = OllamaLLM(model="qwen:0.5b",temperature=0.7)

def basic_llm_demo():
    print("基础LLM调用示例:")
    response = llm.invoke("介绍一下Cusor")
    print(response)
    print("-"*50)

def basic_chain_demo():
    print("基础Chain调用示例:")
    prompt = PromptTemplate.from_template("你好，{name}")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"name": "Cusor"})
    print(response)
    print("-"*50)

if __name__ == "__main__":
    basic_llm_demo()
    basic_chain_demo()