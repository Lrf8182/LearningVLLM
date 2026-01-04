from typing import List, Optional
from pydantic import BaseModel, Field

# ==========================================
# 第一步：用 Pydantic 定义"数据的形状"
# ==========================================


# 定义单个活动
class Activity(BaseModel):
    name: str = Field(description="活动的名称或地点的名字")
    category: str = Field(description="类别，例如：景点、餐饮、交通、购物")
    is_must_visit: bool = Field(description="根据文本判断用户是否强烈表达了'一定要去'的意愿")


# 定义整个行程单
class TripPlan(BaseModel):
    destination: str = Field(description="旅游的目的地城市")
    start_date: str = Field(description="开始日期，格式 YYYY-MM-DD")
    end_date: str = Field(description="结束日期，格式 YYYY-MM-DD")
    arrival_airport: Optional[str] = Field(None, description="到达的机场代码或名称")
    # 嵌套结构！这是 Pydantic 最强的地方
    activities: List[Activity] = Field(description="所有提到的活动列表")
    budget_note: Optional[str] = Field(None, description="关于预算的备注")


# ==========================================
# 第二步：模拟 LLM 的工作 (Extraction)
# ==========================================

# 在真实的 LangChain 代码中，你会这样写：
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# structured_llm = llm.with_structured_output(TripPlan) # 关键：把 Pydantic 类传进去
# result = structured_llm.invoke(user_input_text)

# 为了演示，我们手动模拟 LLM 解析后的数据字典
# 这就是 LLM "看到" Pydantic 定义后，努力生成的 JSON 数据
mock_llm_response_data = {
    "destination": "New York",
    "start_date": "2026-01-08",
    "end_date": "2026-01-12",
    "arrival_airport": "Newark (EWR)",
    "activities": [
        {"name": "The Friends Experience", "category": "景点", "is_must_visit": True},
        {"name": "Friends Apartment Exterior", "category": "景点", "is_must_visit": True},
        {
            "name": "Central Park",
            "category": "景点",
            "is_must_visit": False,  # 用户只说"走走"，语气没那么强烈
        },
    ],
    "budget_note": "稍微控制一下",
}

# ==========================================
# 第三步：数据验证与对象化
# ==========================================

try:
    # 将字典转化为 Pydantic 对象
    trip = TripPlan(**mock_llm_response_data)

    print(f"✅ 解析成功！目的地: {trip.destination}")
    print(f"📅 时间: {trip.start_date} -> {trip.end_date}")

    print("\n📝 待办事项列表:")
    for item in trip.activities:
        # 这里可以使用 Python 对象的方式访问属性，非常舒服
        status = "[必去!]" if item.is_must_visit else "[选去]"
        print(f" - {status} {item.name} ({item.category})")

    # 如果你想把它存入 MongoDB 或发给前端，一键转字典
    # print(trip.model_dump())

except Exception as e:
    print(f"解析失败: {e}")
