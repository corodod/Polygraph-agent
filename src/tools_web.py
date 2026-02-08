#src/tools_web.py
from __future__ import annotations
from typing import Any, Dict, List
from tavily import TavilyClient

class WebSearchTool:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 7) -> List[Dict[str, Any]]:
        res = self.client.search(query=query, max_results=max_results)
        out: List[Dict[str, Any]] = []
        for r in res.get("results", []):
            out.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content") or r.get("snippet"),
            })
        return out
