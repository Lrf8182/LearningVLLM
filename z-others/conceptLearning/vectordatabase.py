import chromadb

# 1. 初始化客户端 (数据存在内存中，重启消失。也可以设为存到硬盘)
client = chromadb.Client()

# 2. 创建一个集合 (类似 SQL 里的 Table)
# Chroma 默认会使用一个内置的轻量级 Embedding 模型 (all-MiniLM-L6-v2)
# 你不需要自己手动把文本转向量，它在后台帮你做了
collection = client.create_collection(name="my_friends_knowledge_base")

# 3. 添加数据 (Upsert)
collection.add(
    documents=[
        "Ross Geller is a paleontologist.",
        "Joey Tribbiani is an actor.",
        "Monica Geller is a chef.",
        "Chandler Bing works in statistical analysis and data reconfiguration.",
    ],
    metadatas=[{"role": "main"}, {"role": "main"}, {"role": "main"}, {"role": "main"}],
    ids=["id1", "id2", "id3", "id4"],
)

# --- 此时，Chroma 已经在后台构建了 HNSW 索引 ---

# 4. 语义搜索
# 用户问的是自然语言，数据库自动转向量并搜索
results = collection.query(
    query_texts=["Who loves cooking?"],  # 问题：谁喜欢做饭？
    n_results=1,  # 只返回最相似的那一个
)

print(results)
