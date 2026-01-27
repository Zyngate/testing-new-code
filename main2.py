from fastapi.responses import HTMLResponse
from fastapi import FastAPI

app = FastAPI()

@app.get("/demo", response_class=HTMLResponse)
def demo():
    with open("task_workflow.html", "r", encoding="utf-8") as f:
        return f.read()
