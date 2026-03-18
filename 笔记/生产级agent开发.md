# AI Agent 30天从入门到精通 · 第3周：Agent 进阶与实战（可直接执行详细任务）
> 目标：从「基础 Agent 调用工具」升级到「生产级 Agent 应用」，掌握复杂场景落地能力（多模型切换、工具扩展、错误处理、可视化）。
> 适配：基于第2周 LangChain 基础，无缝衔接，每天 2～3 小时，代码全部可直接运行。

## 第3周总览（Day15～Day21）
- Day15：多模型集成（Ollama 切换 Llama3/Qwen/DeepSeek）
- Day16：高级 Tool 开发（动态参数、异步工具、工具优先级）
- Day17：Agent 错误处理与容错机制（重试、降级、日志）
- Day18：Agent 与外部数据交互（文件读写、数据库查询）
- Day19：Agent 可视化与调试（LangSmith 监控、日志分析）
- Day20：复杂场景实战（视频剪辑全流程 Agent）
- Day21：第3周项目：**可部署、高容错、多模型的视频处理 AI Agent**

---

## 每日详细任务（可直接执行版）
### Day15：多模型集成（摆脱单一模型限制）
#### 核心目标
掌握 LangChain 切换不同本地/云端模型，解决单一模型能力不足问题。

#### 任务步骤
1. **拉取多版本 Ollama 模型**（30分钟）
   ```bash
   # 拉取 Llama3（8B，效果更好）
   ollama pull llama3:8b
   # 拉取 DeepSeek（代码能力强）
   ollama pull deepseek-coder:6.7b
   # 拉取 Qwen2（最新版通义千问）
   ollama pull qwen2:7b
   ```

2. **多模型切换代码**（40分钟）
   ```python
   # day15_multi_model.py
   from langchain_ollama import OllamaLLM

   # 定义模型字典，一键切换
   MODEL_MAP = {
       "qwen": OllamaLLM(model="qwen2:7b", temperature=0.7),
       "llama3": OllamaLLM(model="llama3:8b", temperature=0.5),
       "deepseek": OllamaLLM(model="deepseek-coder:6.7b", temperature=0.1)
   }

   # 测试不同模型的能力
   def test_multi_model():
       # 1. Qwen2：通用对话
       llm_qwen = MODEL_MAP["qwen"]
       print("=== Qwen2 通用对话 ===")
       print(llm_qwen.invoke("写一段视频剪辑的操作说明"))

       # 2. DeepSeek：代码生成
       llm_deepseek = MODEL_MAP["deepseek"]
       print("\n=== DeepSeek 代码生成 ===")
       print(llm_deepseek.invoke("写一个Python函数，裁剪视频并添加字幕"))

       # 3. Llama3：逻辑推理
       llm_llama3 = MODEL_MAP["llama3"]
       print("\n=== Llama3 逻辑推理 ===")
       print(llm_llama3.invoke("用户要求先裁剪视频再加字幕，步骤应该是什么？"))

   if __name__ == "__main__":
       test_multi_model()
   ```

3. **模型动态选择 Agent**（50分钟）
   ```python
   # 核心逻辑：根据用户指令类型自动选模型
   def select_model_by_task(task):
       if "代码" in task or "函数" in task:
           return MODEL_MAP["deepseek"]
       elif "推理" in task or "步骤" in task:
           return MODEL_MAP["llama3"]
       else:
           return MODEL_MAP["qwen"]
   ```

#### 今日验收
- 能一键切换 3 种不同模型；
- Agent 能根据任务类型自动选择最优模型。

---

### Day16：高级 Tool 开发（解决复杂工具调用问题）
#### 核心目标
开发生产级工具（动态参数、异步、优先级），解决基础 Tool 灵活性不足问题。

#### 任务步骤
1. **动态参数 Tool**（40分钟）
   ```python
   # day16_advanced_tool.py
   from langchain_core.tools import BaseTool
   from langchain_core.pydantic_v1 import BaseModel, Field, validator
   import asyncio

   # 1. 动态参数校验（支持可选参数）
   class VideoEditInput(BaseModel):
       path: str = Field(description="视频路径")
       start: float = Field(description="开始时间，可选", default=0.0)
       end: float = Field(description="结束时间，可选", default=None)
       speed: float = Field(description="倍速，可选", default=1.0)
       subtitle: str = Field(description="字幕，可选", default=None)

       # 自定义校验规则
       @validator("speed")
       def speed_must_be_positive(cls, v):
           if v <= 0:
               raise ValueError("倍速必须大于0")
           return v

   # 2. 异步Tool（提升并发效率）
   class AsyncVideoEditTool(BaseTool):
       name = "async_video_edit"
       description = "异步视频编辑工具，支持裁剪、倍速、加字幕"
       args_schema = VideoEditInput
       async def _arun(self, path: str, start: float=0.0, end: float=None, speed: float=1.0, subtitle: str=None):
           """异步执行视频编辑"""
           # 模拟异步处理（替换为真实视频函数）
           await asyncio.sleep(2)
           result = [f"处理视频：{path}"]
           if start > 0 or end:
               result.append(f"裁剪：{start}→{end}秒")
           if speed != 1.0:
               result.append(f"倍速：{speed}x")
           if subtitle:
               result.append(f"加字幕：{subtitle}")
           return "; ".join(result)

   # 3. 工具优先级（Agent 优先调用高优先级工具）
   def get_tools_with_priority():
       tools = [
           AsyncVideoEditTool(),  # 高优先级
           add_tool,  # 低优先级
       ]
       # 给工具加优先级属性
       tools[0].priority = 1
       tools[1].priority = 10
       return sorted(tools, key=lambda x: x.priority)
   ```

2. **测试异步工具调用**（30分钟）
   ```python
   async def test_async_tool():
       tool = AsyncVideoEditTool()
       result = await tool.arun(path="001.mp4", start=5.0, speed=2.0, subtitle="AI Agent")
       print(result)

   if __name__ == "__main__":
       asyncio.run(test_async_tool())
   ```

#### 今日验收
- Tool 支持可选参数和自定义校验；
- 能异步调用工具；
- Agent 能按优先级选择工具。

---

### Day17：Agent 错误处理与容错机制
#### 核心目标
让 Agent 面对错误不崩溃（工具调用失败、参数错误、模型返回异常）。

#### 任务步骤
1. **错误捕获与重试**（40分钟）
   ```python
   # day17_error_handling.py
   from langchain_core.tools import ToolException
   from langchain.agents import AgentExecutor, create_react_agent
   from tenacity import retry, stop_after_attempt, wait_exponential

   # 1. 工具错误定义
   def risky_video_cut(path, start, end):
       if start >= end:
           raise ToolException(f"参数错误：start({start}) ≥ end({end})", tool_name="video_cut")
       return f"裁剪成功：{path}"

   # 2. 重试装饰器
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
   def call_tool_with_retry(tool, params):
       try:
           return tool.invoke(params)
       except ToolException as e:
           print(f"工具调用失败：{e}，重试中...")
           raise

   # 3. Agent 容错配置
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,
       handle_parsing_errors=True,  # 解析错误容错
       max_iterations=5,  # 最大重试次数
       return_intermediate_steps=True,  # 返回中间步骤，方便调试
       # 自定义错误处理
       on_error=lambda e: f"处理失败：{str(e)}，请检查参数后重试"
   )
   ```

2. **日志记录**（30分钟）
   ```python
   import logging

   # 配置日志
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
       handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()]
   )
   logger = logging.getLogger("AI_Agent")

   # 封装Agent调用，添加日志
   def safe_agent_call(agent, input):
       try:
           logger.info(f"接收指令：{input}")
           result = agent.invoke({"input": input})
           logger.info(f"处理结果：{result['output']}")
           return result
       except Exception as e:
           logger.error(f"处理失败：{str(e)}", exc_info=True)
           return {"output": f"抱歉，处理失败：{str(e)}"}
   ```

#### 今日验收
- Agent 工具调用失败能自动重试；
- 错误信息会记录到日志文件；
- 面对参数错误不会崩溃，返回友好提示。

---

### Day18：Agent 与外部数据交互
#### 核心目标
让 Agent 能读写文件、查询数据库，对接真实业务数据。

#### 任务步骤
1. **文件读写工具**（40分钟）
   ```python
   # day18_data_interaction.py
   @tool
   def read_file_tool(file_path: str) -> str:
       """读取文本文件内容，支持txt/md格式"""
       try:
           with open(file_path, "r", encoding="utf-8") as f:
               return f.read()
       except Exception as e:
           return f"读取失败：{str(e)}"

   @tool
   def write_file_tool(file_path: str, content: str) -> str:
       """写入内容到文本文件"""
       try:
           with open(file_path, "w", encoding="utf-8") as f:
               f.write(content)
           return f"写入成功：{file_path}"
       except Exception as e:
           return f"写入失败：{str(e)}"
   ```

2. **数据库查询工具（SQLite）**（50分钟）
   ```python
   import sqlite3

   # 初始化数据库
   def init_db():
       conn = sqlite3.connect("video_agent.db")
       cursor = conn.cursor()
       # 创建视频任务表
       cursor.execute("""
       CREATE TABLE IF NOT EXISTS video_tasks (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           video_path TEXT,
           operation TEXT,
           status TEXT,
           create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       )
       """)
       conn.commit()
       conn.close()

   @tool
   def query_video_tasks(status: str = None) -> str:
       """查询视频任务，status可选：pending/finished/failed"""
       conn = sqlite3.connect("video_agent.db")
       cursor = conn.cursor()
       if status:
           cursor.execute("SELECT * FROM video_tasks WHERE status = ?", (status,))
       else:
           cursor.execute("SELECT * FROM video_tasks")
       results = cursor.fetchall()
       conn.close()
       return "\n".join([f"任务{row[0]}：{row[1]} - {row[2]} - {row[3]}" for row in results])

   @tool
   def add_video_task(video_path: str, operation: str) -> str:
       """添加视频任务到数据库"""
       conn = sqlite3.connect("video_agent.db")
       cursor = conn.cursor()
       cursor.execute("""
       INSERT INTO video_tasks (video_path, operation, status)
       VALUES (?, ?, 'pending')
       """, (video_path, operation))
       conn.commit()
       conn.close()
       return f"任务添加成功，ID：{cursor.lastrowid}"
   ```

#### 今日验收
- Agent 能读取视频任务配置文件；
- 能将处理结果写入文件；
- 能查询/添加数据库中的视频任务。

---

### Day19：Agent 可视化与调试（LangSmith）
#### 核心目标
用 LangSmith 监控 Agent 运行过程，快速定位问题。

#### 任务步骤
1. **LangSmith 配置**（20分钟）
   ```bash
   # 安装依赖
   pip install langsmith
   ```
   创建 `.env` 文件，添加：
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=你的LangSmith API Key（官网注册获取）
   LANGCHAIN_PROJECT=AI_Agent_Video_Editor
   ```

2. **启用 LangSmith 追踪**（30分钟）
   ```python
   # day19_debug.py
   import os
   from dotenv import load_dotenv
   load_dotenv()

   # 初始化LangSmith
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

   # 运行之前的Agent，自动上传追踪数据
   def test_langsmith():
       agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
       result = agent_executor.invoke({"input": "裁剪001.mp4从5到10秒，并记录到数据库"})
       print(result)
   ```

3. **日志分析工具**（40分钟）
   ```python
   def analyze_agent_log(log_file: str = "agent.log"):
       """分析Agent日志，统计成功/失败次数"""
       success_count = 0
       fail_count = 0
       errors = []
       
       with open(log_file, "r", encoding="utf-8") as f:
           for line in f:
               if "处理成功" in line:
                   success_count += 1
               elif "处理失败" in line:
                   fail_count += 1
                   errors.append(line.strip())
       
       report = f"""
       Agent 运行报告：
       - 成功次数：{success_count}
       - 失败次数：{fail_count}
       - 失败详情：
       {chr(10).join(errors[:5])}
       """
       return report
   ```

#### 今日验收
- 能在 LangSmith 控制台看到 Agent 的思考/行动/观察全过程；
- 能通过日志分析工具统计 Agent 运行情况。

---

### Day20：复杂场景实战（视频剪辑全流程 Agent）
#### 核心目标
开发能处理完整视频剪辑流程的 Agent（解析需求→拆分任务→调用工具→保存结果→记录日志）。

#### 任务步骤
1. **全流程 Agent 代码**（60分钟）
   ```python
   # day20_video_agent.py
   from langchain_core.prompts import PromptTemplate
   from langchain.agents import create_react_agent, AgentExecutor
   from langchain_ollama import OllamaLLM
   import logging

   # 初始化日志
   logger = logging.getLogger("VideoAgent")

   # 1. 定义所有工具
   tools = [
       read_file_tool,
       write_file_tool,
       add_video_task,
       query_video_tasks,
       AsyncVideoEditTool()
   ]

   # 2. 全流程提示词
   prompt = PromptTemplate.from_template("""
   你是专业的视频剪辑AI Agent，需要完成以下完整流程：
   1. 解析用户的视频剪辑需求
   2. 拆分具体操作步骤（裁剪/倍速/加字幕）
   3. 调用对应的视频编辑工具
   4. 将任务记录到数据库
   5. 将处理结果写入日志文件
   
   可用工具：{tools}
   格式要求：
   思考：分析需求，决定下一步操作
   行动：工具名称
   行动输入：{"参数名":值}
   观察：工具执行结果
   最终答案：总结处理结果，包含任务ID和文件路径
   
   用户指令：{input}
   """)

   # 3. 创建全流程Agent
   llm = OllamaLLM(model="llama3:8b", temperature=0.3)
   agent = create_react_agent(llm, tools, prompt)
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,
       handle_parsing_errors=True,
       max_iterations=10
   )

   # 4. 全流程测试
   def test_full_video_agent():
       # 复杂指令
       complex_input = """
       处理视频001.mp4：
       1. 裁剪从3秒到15秒
       2. 设置2倍速播放
       3. 添加字幕「AI Agent 视频剪辑」
       4. 将处理结果保存到video_result.txt
       5. 记录任务到数据库
       """
       result = safe_agent_call(agent_executor, complex_input)
       print(f"最终结果：{result['output']}")

   if __name__ == "__main__":
       init_db()  # 初始化数据库
       test_full_video_agent()
   ```

#### 今日验收
- Agent 能处理包含多个步骤的复杂视频剪辑需求；
- 自动完成工具调用、数据记录、文件保存全流程。

---

### Day21：第3周项目：可部署的视频处理 AI Agent
#### 最终交付物
1. 一个完整的 `video_agent.py` 文件；
2. 支持多模型切换、异步工具、错误处理、数据持久化；
3. 包含日志、数据库、配置文件；
4. 可直接部署运行。

#### 项目结构
```
AI_Agent_LangChain/
├── video_agent.py       # 主程序
├── agent.log            # 运行日志
├── video_agent.db       # SQLite数据库
├── video_result.txt     # 处理结果
├── .env                 # 配置文件
└── requirements.txt     # 依赖列表
```

#### 部署测试（30分钟）
```bash
# 生成依赖列表
pip freeze > requirements.txt

# 运行Agent
python video_agent.py
```

#### 验收标准
- 输入任意复杂视频剪辑指令，Agent 能自动完成所有步骤；
- 即使出现参数错误/工具调用失败，也能优雅处理；
- 所有操作都有日志记录，可追溯。

---

## 第3周核心知识点总结
1. **多模型策略**：根据任务类型动态选择最优模型（通用/代码/推理）；
2. **生产级 Tool**：支持动态参数、异步调用、优先级排序；
3. **容错机制**：错误捕获、自动重试、日志记录，保证 Agent 稳定性；
4. **数据交互**：对接文件系统和数据库，实现业务数据持久化；
5. **可观测性**：LangSmith 监控 + 日志分析，快速定位问题；
6. **端到端实战**：从需求解析到结果输出的完整 Agent 流程。

---

### 进阶建议
- 第3周完成后，可尝试将 Agent 封装为 API 接口（FastAPI）；
- 对接前端页面，实现可视化操作界面；
- 扩展更多视频处理工具（转码、加水印、合并）。
