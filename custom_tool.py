# 注释：Day10主题是自定义LangChain Tool
from langchain_core.tools import BaseTool, tool  # 导入BaseTool类和tool装饰器
from pydantic import BaseModel, Field  # 导入Pydantic模型用于参数校验
from langchain_ollama import OllamaLLM  # 导入OllamaLLM
from langchain_core.prompts import PromptTemplate  # 导入提示词模板
from langchain_core.output_parsers import StrOutputParser  # 导入字符串解析器

# 方式1：用@tool装饰器定义工具，最简单的方式
@tool  # 装饰器会自动把函数转成Tool对象，函数文档字符串会作为工具描述
def add_tool(a: int, b: int) -> int:
    """加法工具：计算两个整数的和"""  # 工具描述，Agent会根据这个决定是否调用
    return a + b  # 工具执行逻辑

@tool
def subtract_tool(a: int, b: int) -> int:
    """减法工具：计算a减去b的结果"""
    return a - b

# 方式2：用类继承BaseTool定义工具，适合复杂工具
# 定义工具的输入参数模型，指定每个参数的描述，用于Agent理解如何传参
class VideoCutInput(BaseModel):
    path: str = Field(description="视频文件路径，如0001.mp4")  # 视频路径参数
    start: float = Field(description="裁剪开始时间（秒）")  # 开始时间参数
    end: float = Field(description="裁剪结束时间（秒）")  # 结束时间参数

# 定义视频裁剪工具类，继承BaseTool
class VideoCutTool(BaseTool):
    name = "video_cut"  # 工具名称，Agent会用这个名称调用
    description = "视频裁剪工具：裁剪指定时间段的视频"  # 工具描述
    args_schema = VideoCutInput  # 绑定参数模型，自动校验输入

    def _run(self, path: str, start: float, end: float) -> str:  # 工具核心执行方法，必须实现
        """核心执行逻辑，这里是模拟视频裁剪，实际可以替换成真实的视频处理代码"""
        if start >= end:  # 校验开始时间是否小于结束时间
            return f"裁剪失败：开始时间{start}≥结束时间{end}"  # 错误返回
        return f"裁剪成功：{path} 从{start}秒到{end}秒，输出文件：{path}_cut.mp4"  # 成功返回

# 定义测试函数，测试工具是否能正常工作
def test_tools():
    print("=== 测试加法工具 ===")
    add_result = add_tool.invoke({"a": 5, "b": 3})  # 调用加法工具，传入参数a=5，b=3
    print(f"5+3={add_result}")  # 打印结果

    print("=== 测试视频裁剪工具 ===")
    video_tool = VideoCutTool()  # 实例化视频裁剪工具
    video_result = video_tool.invoke({"path": "0001.mp4", "start": 5.0, "end": 10.0})
    print(video_result)  # 21. 打印工具返回结果

if __name__ == "__main__":  # Python入口函数
    test_tools()  # 调用测试函数