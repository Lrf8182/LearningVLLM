from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. 定义工具 (Tools)
# 使用 @tool 装饰器，LangChain 会自动提取函数名、参数类型和 docstring 作为工具描述
# ==========================================


@tool
def google_search(query: str) -> str:
    """
    当用户询问时事新闻、天气、或者世界上的通用知识时，使用此工具搜索 Google。
    """
    # 这里模拟真实的搜索，实际应用中可以调用 Serper 或 Google API
    print(f"\n[Tool Called] 正在调用 Google Search... 查询: {query}")
    if "天气" in query:
        return "Google 搜索结果: 今天旧金山天气晴朗，气温 20 摄氏度。"
    return "Google 搜索结果: Llama 3 是 Meta 发布的最新的开源大语言模型。"


@tool
def read_local_file(filename: str) -> str:
    """
    当用户询问关于'内部会议'、'私有文档'、'项目代码'时，使用此工具读取本地文件。
    """
    print(f"\n[Tool Called] 正在读取本地文件... 文件名: {filename}")
    # 模拟读取文件
    return f"文件 '{filename}' 的内容是：下周一上午 9 点进行全员技术代码审查。"


# 将工具放入列表
tools = [google_search, read_local_file]

# ==========================================
# 2. 初始化本地 Llama 3 (通过 LangChain 的 ChatOpenAI)
# 关键点：base_url 指向 vLLM 的地址
# ==========================================

llm = ChatOpenAI(
    model="llama3",  # 对应 vLLM 启动时的 --served-model-name
    openai_api_key="token-123",
    openai_api_base="http://localhost:8000/v1",  # vLLM 的地址
    temperature=0,
)

# ==========================================
# 3. 绑定工具 (Tool Binding)
# 这步操作会把工具的 JSON Schema 注入到 Llama 3 的系统提示词中
# ==========================================

llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 4. 测试场景
# ==========================================


def run_agent(user_query):
    print(f"\n{'=' * 10} 用户提问: {user_query} {'=' * 10}")

    messages = [HumanMessage(content=user_query)]

    # 让 Llama 3 思考并决定
    ai_msg = llm_with_tools.invoke(messages)

    # 检查 AI 是否决定调用工具
    if ai_msg.tool_calls:
        print(f"👉 AI 决定调用工具: {ai_msg.tool_calls[0]['name']}")
        print(f"👉 参数: {ai_msg.tool_calls[0]['args']}")

        # --- 在真实的 Agent 循环中，这里会执行工具并将结果返回给 LLM ---
        # --- 这里为了演示清晰，我们只展示到"决策"这一步 ---
    else:
        print("👉 AI 决定直接回答 (不使用工具)")
        print(f"回答: {ai_msg.content}")


# --- 测试 1: 应该触发 Google ---
run_agent("今天旧金山天气怎么样？")

# --- 测试 2: 应该触发本地文件 ---
run_agent("帮我查一下内部会议记录里下周一有什么安排？")

# --- 测试 3: 通用聊天 ---
run_agent("你好，讲个笑话。")
