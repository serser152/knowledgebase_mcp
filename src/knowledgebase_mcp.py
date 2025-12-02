"""
    Модуль для запуска сервера MCP.
"""
from mcp.server.fastmcp import FastMCP
from faiss_base import KnowledgeBase

mcp = FastMCP("Planner", host="0.0.0.0", port=8000)

# load docs
k = KnowledgeBase()
k.load_docs()
print(f'Loaded {k.index.ntotal} docs')

@mcp.tool()
async def query_knowledgebase(query: str) -> str:
    """Запросить данные из базы знаний.
    Args: query (str) - запрос, интересующая информация
    Returns: str: ответ
    """
    print("query:", query)
    results = k.find_similar(query)
    print("Results:",results)

    res = '\n-----------\n'.join([i['txt'] for i in results])
    print("Res:",res)

    return res

if __name__ == "__main__":
    mcp.run(transport="streamable-http")