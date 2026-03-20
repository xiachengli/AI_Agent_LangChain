
from langchain_core.tools import tool, StructuredTool

# 自定义工具方式1：使用装饰器@tool
@tool(name_or_callable="add_numbers", description="计算两个数字的和", return_direct=True)
def add_numbers(a, b):
    """计算两个数字的和"""
    return a + b


print(f"(name = {add_numbers.name}")  #工具名称
print(f"description = {add_numbers.description}")  #工具描述
print(f"args_schema = {add_numbers.args_schema}")   #工具参数模式
print(f"args = {add_numbers.args}")   #工具参数列表
print(f"return_direct = {add_numbers.return_direct}")   #是否直接返回结果，默认为False，仅与Agent相关   表示返回一个包含工具调用信息的字典；如果设置为True，则直接返回工具的输出结果。

# 函数调用
#print(add_numbers(3, 2))

# 调用工具
print(add_numbers.invoke({"a": 3, "b": 2}))

# 自定义工具方式2：StructuredTool
def my_tool(x: int, y: int) -> int:
    """计算两个整数的乘积"""
    return x * y

my_tool_structured = StructuredTool.from_function(
    func=my_tool,
    name="multiply_numbers",
    description="计算两个整数的乘积",       
    return_direct=True
)

print(f"(name = {my_tool_structured.name}")  #工具名称
print(f"description = {my_tool_structured.description}")  #工具描述
print(my_tool_structured.invoke({"x": 4, "y": 5}))  #调用工具

