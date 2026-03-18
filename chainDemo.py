# chainDemo.py - LangChain Chain 入门演示（Ollama + 本地环境版）
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 1. 定义提示词模板
prompt = PromptTemplate(
    input_variables=["topic"],
    template="给我写一段关于 {topic} 的学习总结，要求简洁明了。"
)

# 2. 加载本地 Ollama 模型（qwen:0.5b）
# 确保已经运行过: ollama pull qwen:0.5b
llm = OllamaLLM(model="qwen:0.5b", temperature=0.7)

# 3. 构建 Chain（LCEL 标准写法：推荐链式调用）
# 流程：输入 -> 提示词模板 -> 模型 -> 输出解析
chain = prompt | llm | StrOutputParser()

# 4. 执行 Chain
if __name__ == "__main__":
    topic = "LangChain 核心框架"
    print(f"正在生成关于 [{topic}] 的总结...\n")
    
    # 调用模型
    result = chain.invoke({"topic": topic})
    
    # 打印结果
    print("=== 生成结果 ===")
    print(result)