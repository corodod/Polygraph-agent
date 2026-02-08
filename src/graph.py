#src/graph.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END
from src.tools_web import WebSearchTool
from src.tools_calc import eval_equality
from src.openrouter_client import llm_json
from src.prompts import (
    SYSTEM_ROUTER, USER_ROUTER,
    SYSTEM_QUERY_REWRITE, USER_QUERY_REWRITE,
    SYSTEM_JUDGE, USER_JUDGE_WEB, USER_JUDGE_CALC
)

class PolyState(TypedDict, total=False):
    text: str
    route: Literal["web", "calc"]
    calc_expression: Optional[str]
    calc_verdict: Optional[bool]
    query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]
    verdict: Optional[bool]

def build_graph(web: WebSearchTool):
    
    def router(state: PolyState) -> PolyState:
        obj = llm_json(
            SYSTEM_ROUTER,
            USER_ROUTER.format(text=state["text"]),
            max_tokens=200
        )

        route = obj.get("route")
        if route not in ("web", "calc"):
            # эвристика: если похоже на математику — calc, иначе web
            t = state["text"]
            mathy = (
                any(ch in t for ch in "+-*/=()")
                or any(c.isdigit() for c in t)
                or ("sin" in t.lower())
                or ("cos" in t.lower())
            )
            route = "calc" if mathy else "web"

        return {
            "route": route,
            "calc_expression": obj.get("calc_expression"),
        }



    def math_parser(state: PolyState) -> PolyState:
        expr = state["text"]   # <-- только оригинал, без LLM
        verdict = eval_equality(expr)
        print(f"[MATH] expr={expr!r} => {verdict}", flush=True)
        return {"calc_verdict": verdict}

    
    def query_rewriter(state: PolyState) -> PolyState:
        obj = llm_json(
            SYSTEM_QUERY_REWRITE,
            USER_QUERY_REWRITE.format(text=state["text"]),
            max_tokens=200
        )
        q = obj.get("query")
        if not q or not isinstance(q, str):
            q = state["text"]
        return {"query": q}


    def web_node(state: PolyState) -> PolyState:
        q = state.get("query") or state["text"]
        results = web.search(q, max_results=7)
        return {"search_results": results}
    
    # def judge(state: PolyState) -> PolyState:
    #     if state.get("route") == "calc":
    #         obj = llm_json(
    #             SYSTEM_JUDGE,
    #             USER_JUDGE_CALC.format(text=state["text"], calc_verdict=state.get("calc_verdict")),
    #             max_tokens=120
    #         )
    #         print(f"[JUDGE-CALC] calc_verdict={state.get('calc_verdict')}", flush=True)
    #     else:
    #         evidence = state.get("search_results") or []
    #         obj = llm_json(
    #             SYSTEM_JUDGE,
    #             USER_JUDGE_WEB.format(text=state["text"], evidence=json.dumps(evidence, ensure_ascii=False)),
    #             max_tokens=250
    #         )

    #     v = obj.get("verdict")
    #     if isinstance(v, bool):
    #         return {"verdict": v}

    #     # fallback: если модель вернула "True"/"False" строкой
    #     if isinstance(v, str):
    #         vv = v.strip().lower()
    #         if vv in ("true", "1", "yes"):
    #             return {"verdict": True}
    #         if vv in ("false", "0", "no"):
    #             return {"verdict": False}

    #     # fallback: если вообще странный ответ
    #     return {"verdict": False}
    def judge(state: PolyState) -> PolyState:
        if state.get("route") == "calc":
            v = bool(state.get("calc_verdict"))
            print(f"[JUDGE-CALC] final={v} calc_verdict={state.get('calc_verdict')}", flush=True)
            return {"verdict": v}

        evidence = state.get("search_results") or []
        obj = llm_json(
            SYSTEM_JUDGE,
            USER_JUDGE_WEB.format(text=state["text"], evidence=json.dumps(evidence, ensure_ascii=False)),
            max_tokens=250
        )
        v = obj.get("verdict")
        return {"verdict": bool(v) if isinstance(v, bool) else False}

    
    def route_decision(state: PolyState) -> str:
    # Никогда не падаем с KeyError
        r = state.get("route")
        return "math_parser" if r == "calc" else "query_rewriter"

    g = StateGraph(PolyState)
    g.set_entry_point("router")

    g.add_node("router", router)
    g.add_node("math_parser", math_parser)
    g.add_node("query_rewriter", query_rewriter)
    g.add_node("web_node", web_node)
    g.add_node("judge", judge)

    g.add_conditional_edges("router", route_decision, {
        "math_parser": "math_parser",
        "query_rewriter": "query_rewriter",
    })
    g.add_edge("math_parser", "judge")
    g.add_edge("query_rewriter", "web_node")
    g.add_edge("web_node", "judge")
    g.add_edge("judge", END)

    return g.compile()
