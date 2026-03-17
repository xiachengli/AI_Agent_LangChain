# 导入 Ollama 模型
from langchain_ollama import OllamaLLM

# 加载本地的 qwen 模型
llm = OllamaLLM(model="qwen:0.5b")

# 调用模型（和 OpenAI 的 invoke 方法完全一样）
response = llm.invoke("介绍一下 LangChain用中文")
print(response)