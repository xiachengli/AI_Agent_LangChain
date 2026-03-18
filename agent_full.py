# Day14：最终成品 - 可对话+记忆+多工具AI Agent
from langchain_ollama import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor  # 1. 导入Agent创建工具
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==================== 1. 工具定义 ====================
def add(a: int, b: int) -> str:  # 2. 加法工具函数
    return f"{a}+{b}={a+b}"

def subtract(a: int, b: int) -> str:  # 3. 减法工具函数
    return f"{a}-{b}={a-b}"

def video_cut(path: str, start: float, end: float) -> str:  # 4. 视频裁剪工具函数
    if start >= end:
        return f"裁剪失败：{start}≥{end}"
    return f"裁剪成功：{path}({start}→{end}秒) → {path}_cut.mp4"

def add_subtitle(path: str, text: str) -> str:  # 5. 视频加字幕工具函数
    return f"加字幕成功：{path} → 字幕「{text}」→ {path}_with_sub.mp4"

tools = [  # 6. 工具列表：将所有工具包装成LangChain的Tool对象
    Tool(name="加法", func=lambda a,b: add(int(a),int(b)), description="计算两个整数的和"),
    Tool(name="减法", func=lambda a,b: subtract(int(a),int(b)), description="计算两个整数的差"),
    Tool(name="视频裁剪", func=lambda p,s,e: video_cut(p,float(s),float(e)), description="裁剪视频时间段"),
    Tool(name="视频加字幕", func=lambda p,t: add_subtitle(p,t), description="给视频加字幕")
]

# ==================== 2. Agent配置 ====================
#llm = OllamaLLM(model="qwen:14b", temperature=0.1)  # 7. 加载模型，temperature=0.1减少随机性，适合工具调用

llm = OllamaLLM(
    model="qwen3.5:9b", 
    temperature=0.7,
    num_ctx=4096,  # 上下文长度
    top_p=0.9,
)

# ReAct提示词（带记忆）
prompt = PromptTemplate.from_template("""  # 8. Agent的核心提示词模板
你是有记忆的AI Agent，能执行数学计算和视频剪辑任务，会记住用户之前的对话。
历史对话：{chat_history}  # 9. 记忆占位符：填充历史对话
可用工具：{tools}  # 10. 工具列表占位符：告诉Agent有哪些工具
格式要求：  # 11. ReAct思考步骤格式
思考：分析用户指令，决定用什么工具
行动：工具名称
行动输入：{"参数名":值}
观察：工具执行结果
最终答案：总结结果

用户指令：{input}  # 12. 用户当前输入
""")

# 创建Agent
agent = create_react_agent(llm, tools, prompt)  # 13. 用ReAct逻辑创建Agent
agent_executor = AgentExecutor(  # 14. Agent执行器：管理Agent的思考-行动循环
    agent=agent,
    tools=tools,
    verbose=True,  # 15. 打印思考过程，方便调试
    handle_parsing_errors=True  # 16. 容错：解析错误时继续运行
)

# ==================== 3. 记忆系统 ====================
chat_history = InMemoryChatMessageHistory()  # 17. 初始化内存记忆
agent_with_history = RunnableWithMessageHistory(  # 18. 绑定记忆到Agent
    agent_executor,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# ==================== 4. 对话入口 ====================
def chat_agent():
    print("=== AI Agent v2.0（可对话+工具+记忆）===")
    print("支持指令：加法/减法/视频裁剪/视频加字幕，输入「退出」结束对话\n")
    session_id = "final_agent_session"  # 19. 固定会话ID，确保记忆连贯
    
    while True:  # 20. 无限循环，实现多轮对话
        user_input = input("你：")  # 21. 获取用户输入
        if user_input.lower() in ["退出", "exit", "quit"]:  # 22. 退出条件判断
            print("Agent：再见！")
            break  # 23. 跳出循环，结束对话
        
        # 执行带记忆的Agent
        result = agent_with_history.invoke(  # 24. 调用Agent处理用户输入
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"Agent：{result['output']}\n")  # 25. 打印Agent最终回复

if __name__ == "__main__":
    chat_agent()  # 26. 启动对话入口