# 注释：Day9主题是提示词工程和结构化输出
from langchain_ollama import OllamaLLM  # 导入OllamaLLM
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate  # 导入普通和示例提示词模板
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser  # 导入JSON和字符串输出解析器
from pydantic import BaseModel, Field  # 直接从 pydantic 导入

# 加载模型
llm = OllamaLLM(
    model="qwen3.5:9b", 
    temperature=0.7,
    num_ctx=4096,  # 上下文长度
    top_p=0.9,
)

# 定义一个Pydantic模型，规定JSON输出的结构：必须有answer和reason两个字段，Field是字段描述
class AnswerSchema(BaseModel):
    answer: str = Field(description="问题的答案")
    reason: str = Field(description="答案的理由")

# 定义函数，演示JSON结构化输出
def json_format_demo():
    print("=== JSON结构化输出 ===")  # 打印标题
    parser = JsonOutputParser(pydantic_object=AnswerSchema)  # 创建JSON解析器，指定用AnswerSchema验证结构
    # 创建提示词模板，partial_variables是固定参数，这里传入解析器的格式说明
    prompt = PromptTemplate(
        input_variables=["question"],  # 动态变量是question
        template="""
你是AI助手，必须严格按照指定格式输出JSON。
格式要求：{format_instructions}  # 这里会替换成parser生成的JSON格式说明

问题：{question}  # 这里替换成用户的问题
        """,
        partial_variables={"format_instructions": parser.get_format_instructions()}  # 固定传入格式说明
    )
    chain = prompt | llm | parser  # 构建Chain：提示词→模型→JSON解析
    result = chain.invoke({"question": "LangChain的核心价值是什么？"})  # 传入问题
    print(f"答案：{result['answer']}")  # 从JSON结果中取answer字段
    print(f"理由：{result['reason']}")  # 取reason字段
    print("-" * 50)  # 分隔线

# 定义函数，演示FewShot示例提示词
def few_shot_demo():
    """
    Demonstrates how to use a few-shot prompt template with LangChain to guide an LLM's response format.

    The function:
    - Prints a title indicating a few-shot prompt example.
    - Defines example input-output pairs to show desired response style.
    - Sets up a prompt template for formatting each example.
    - Combines examples, template, prefix, and suffix into a FewShotPromptTemplate.
    - Constructs a chain with the prompt template, LLM, and output parser.
    - Invokes the chain with a new input ("AI Agent") to generate a model response mimicking the examples.
    - Prints the generated result.

    This is useful for instructing language models to answer questions in a specific format by providing sample demonstrations.
    """
    print("=== FewShot示例提示词 ===")  # 打印标题
    # 定义示例列表，每个示例是input和output的对应
    examples = [
        {"input": "LangChain", "output": "LangChain是AI Agent开发框架，支持工具调用、记忆、链等功能"},
        {"input": "Ollama", "output": "Ollama是本地大模型运行工具，支持Llama3、Qwen等模型一键部署"}
    ]
    # 定义示例模板，说明每个示例的格式
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="输入：{input}\n输出：{output}"  # 示例的显示格式
    )
    # 创建FewShot提示词模板，把示例、示例模板、前缀、后缀组合起来
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,  # 传入示例
        example_prompt=example_prompt,  # 示例的格式
        prefix="请按照以下示例格式回答问题：",  # 提示词开头的说明
        suffix="输入：{input}\n输出：",  # 提示词结尾，留出用户输入的位置
        input_variables=["input"]  # 动态变量是input
    )
    chain = few_shot_prompt | llm | StrOutputParser()  # 构建Chain
    result = chain.invoke({"input": "AI Agent"})  # 传入新的输入，让模型模仿示例回答
    print(result)  # 打印结果

# 运行函数
if __name__ == "__main__":
    json_format_demo()  # 演示JSON结构化输出
    few_shot_demo()  # 演示FewShot提示词