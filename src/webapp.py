#src/webapp.py
from __future__ import annotations
import os
from dotenv import load_dotenv

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.tools_web import WebSearchTool
from src.graph import build_graph
import traceback

load_dotenv()

app = FastAPI(title="Polygraph Agent")
templates = Jinja2Templates(directory="templates")

# Инициализируем graph один раз при старте приложения
web = WebSearchTool(api_key=os.environ["TAVILY_API_KEY"])
polygraph = build_graph(web)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "fact": "", "verdict": None, "error": None},
    )


@app.post("/check", response_class=HTMLResponse)
def check_fact(request: Request, fact: str = Form(...)):
    fact = (fact or "").strip()
    if not fact:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "fact": "", "verdict": None, "error": "Введите факт"},
        )

    try:
        result = polygraph.invoke({"text": fact})
        verdict = bool(result["verdict"])
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "fact": fact, "verdict": verdict, "error": None},
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "fact": fact, "verdict": None, "error": tb},
        )


@app.post("/api/check")
def api_check(fact: str = Form(...)):
    """
    API-эндпоинт: удобно для постмана/фронта.
    """
    fact = (fact or "").strip()
    result = polygraph.invoke({"text": fact})
    return {"fact": fact, "verdict": bool(result["verdict"])}
