from langchain_community.tools import tool

@tool
def mul(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

class mathToolkit:
    def get_tools(self):
        return [mul, add]

math_toolkit = mathToolkit()
tools = math_toolkit.get_tools()

for tool in tools:
    print(tool.name)
    print(tool.description)
