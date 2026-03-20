from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_classic import hub
from langchain_core.prompts import PromptTemplate

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全地计算数学表达式
        # 只允许基本的数学运算，避免安全风险
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不安全字符"
        
        # 使用eval计算，但限制在数学表达式范围内
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

llm = ChatOllama(model="qwen2.5:3b", base_url="http://localhost:11434")

# 方法1：Hub 提示（推荐）
try:
    prompt = hub.pull("hwchase17/react")
    print("✅ Hub 提示加载成功")
except:
    # 方法2：手动提示
    prompt = PromptTemplate.from_template("""
    你是一个AI助手，使用工具解决问题。
    {tools}
    
    格式：
    问题：{input}
    {agent_scratchpad}
    """)
    print("✅ 手动提示创建成功")

memory = MemorySaver()
agent = create_react_agent(llm, tools=[calculator], prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=[calculator],
    checkpointer=memory,
    verbose=True,
    handle_parsing_errors=True
)

# 测试
result = agent_executor.invoke(
    {"input": "100+250= ?"},
    config={"configurable": {"thread_id": "test"}}
)
print("✅ 运行成功:", result["output"])

result = agent_executor.invoke(
    {"input": "计算2+2*3等于多少？"},
    config={"configurable": {"thread_id": "test"}}
)
print("✅ 运行成功:", result["output"])