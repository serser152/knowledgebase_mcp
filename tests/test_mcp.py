#!/usr/bin/env python
"""
Agent module
Agent tools and llm initialization
"""

import datetime
import pytest

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(find_dotenv())


@pytest.mark.asyncio
async def test_mcp():
    """Test MCP server with simple ollama model with tools"""
    client = MultiServerMCPClient(
        {
            "knowledgebase": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    print(tools)
    llm = ChatOllama(model="gpt-oss:20b",
                     base_url="http://localhost:11434",
                     system_prompt="Используй только  информацию из базы знаний для ответа на вопросы.")

    llm.bind_tools(tools)

    a = create_agent(llm,
                     tools,
                     )
    # res = await a.ainvoke({'messages': [{'role': 'user', 'content': 'Что есть в базе знаний по поводу конституции?'}]})
    # res = res['messages'][-1].content
    # print(res)
    async for chunk in a.astream({'messages': [{'role': 'user', 'content': 'Что есть в базе знаний по поводу конституции?'}]}, stram_mode="updates"):
        print(chunk)
    print(chunk['model']['messages'][0].content)
    #assert res.find('delete') >= 0
