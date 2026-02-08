#src/main.py
import os
import pandas as pd
from dotenv import load_dotenv

from src.tools_web import WebSearchTool
from src.graph import build_graph

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # корень проекта
DATA_PATH = BASE_DIR / "data" / "questions.csv"

df = pd.read_csv(DATA_PATH)

def main():
    load_dotenv()

    web = WebSearchTool(api_key=os.environ["TAVILY_API_KEY"])
    app = build_graph(web)

    # df = pd.read_csv("data/dataset.csv")
    df = pd.read_csv(DATA_PATH)
    answers = []

    for text in df["texts"].tolist():
        out = app.invoke({"text": text})
        answers.append(bool(out["verdict"]))

    df["answers"] = answers
    df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
