# 笔记：LangChain 本地开发环境搭建（Ollama + 虚拟环境）
## 一、核心目标
搭建无需外网、免费的 LangChain 本地开发环境，使用 Ollama 运行本地大模型（通义千问 qwen:0.5b），避开 OpenAI API 网络/付费问题。

Ollama是一个开源的、轻量级的本地大语言模型运行平台，相当于大模型的"应用商店"+"运行环境"。它让你能像安装手机App一样，在个人电脑上一键部署和运行各种开源大模型。

## 二、环境搭建步骤
### 1. 创建并激活 Python 虚拟环境（Windows PowerShell）
```powershell
# 1. 创建专属虚拟环境（隔离依赖，避免冲突）
python -m venv langchain-env

# 2. 激活虚拟环境（PowerShell 专用命令）
.\langchain-env\Scripts\Activate.ps1
# 激活成功后，终端前缀会显示 (langchain-env)
```

### 2. 安装 LangChain 相关依赖
```powershell
# 安装核心依赖包（适配 Ollama + LangChain v0.2+）
pip install langchain langchain-community python-dotenv langchain-ollama
```
| 包名 | 作用 |
|------|------|
| langchain | LangChain 核心框架 |
| langchain-community | 社区拓展工具/集成 |
| python-dotenv | 加载环境变量（可选） |
| langchain-ollama | LangChain 对接 Ollama 的专用包 |

### 3. 安装并配置 Ollama（本地大模型运行工具）
#### 步骤1：安装 Ollama
- 下载地址：https://ollama.com/
- 安装方式：双击安装包，自动完成（无需手动配置）。

#### 步骤2：拉取本地模型（通义千问 qwen:0.5b）
```powershell
# 方式1：Ollama 自带窗口（推荐，无需环境变量）
# 打开 Windows 开始菜单 → 启动 Ollama → 在窗口中输入：
pull qwen:0.5b

# 方式2：VS Code 终端（需环境变量生效/指定完整路径）
ollama pull qwen:0.5b
# 查看已安装模型
ollama list
#卸载
ollama rm qwen:14b
ollama pull qwen3.5:9b
# 查看模型的详细信息
ollama show qwen3.5:9b
# 旧代码（使用 qwen:14b）
# llm = OllamaLLM(model="qwen:14b", temperature=0.7)

# 新代码（使用 qwen3.5:9b）
llm = OllamaLLM(
    model="qwen3.5:9b", 
    temperature=0.7,
    num_ctx=4096,  # 上下文长度
    top_p=0.9,
)
# （若识别不到命令，重启 VS Code/电脑，或用完整路径：& 'C:\Users\你的用户名\AppData\Local\Programs\Ollama\ollama.exe' pull qwen:0.5b）
```
- 模型特点：qwen:0.5b 是超轻量版通义千问，仅 0.5B 参数，低配电脑也能流畅运行。

#### 步骤3：测试本地模型（终端直接对话）
```powershell
# 启动模型交互式对话
ollama run qwen:0.5b

# 输入任意问题即可测试，例如：
你好，介绍一下 LangChain
```

## 三、关键说明
### 1. 环境隔离优势
- 专用虚拟环境 `langchain-env` 避免与其他项目（如视频剪辑）的依赖冲突；
- Ollama 本地模型无需外网、无需 API Key，完全免费，适合入门学习。

### 2. 避坑提醒
- PowerShell 激活虚拟环境时，若提示权限错误，需先执行：`Set-ExecutionPolicy RemoteSigned`（选择 Y 确认）；
- Ollama 命令识别失败：优先用 Ollama 自带窗口操作，或重启终端/电脑加载环境变量；
- 模型下载慢：可换网络时段，或选择更小的模型（如 qwen:0.5b）。

### 3. 后续衔接
环境搭建完成后，可编写 Python 代码调用本地模型：
```python
from langchain_ollama import OllamaLLM

# 加载本地 qwen:0.5b 模型
llm = OllamaLLM(model="qwen:0.5b")
# 调用模型生成内容
response = llm.invoke("介绍一下 LangChain 的核心功能")
print(response)
```

## 四、总结
1. 虚拟环境确保依赖隔离，避免版本冲突；
2. Ollama 实现本地大模型运行，彻底解决外网 API 问题；
3. qwen:0.5b 轻量易部署，是 LangChain 入门的最优本地模型选择。