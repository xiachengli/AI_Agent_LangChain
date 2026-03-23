


#1 初始化文档嵌入模型
import ollama

response = ollama.embed(
    model='qwen3-embedding',
    input='The sky is blue because of Rayleigh scattering',
)
print(response.embeddings)

#句子向量化

#文档向量化