import json
import asyncio
from typing import Dict, Any
import bleach

from config import logger
from services.ai_service import get_visualize_groq_client

# -------------------------------
# HTML SANITIZATION
# -------------------------------
ALLOWED_TAGS = [
    "div", "h1", "h2", "h3", "h4", "p",
    "table", "thead", "tbody", "tr", "th", "td",
    "ul", "li", "span", "canvas"
]

ALLOWED_ATTRS = {
    "*": ["class", "data-chart"]
}

def sanitize_html(html: str) -> str:
    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        strip=True
    )

# -------------------------------
# HTML BUILDER
# -------------------------------
def build_visualization_html(plan: Dict[str, Any]) -> str:
    html = f"<h2>{plan.get('title', 'Visualization')}</h2>"

    for block in plan.get("blocks", []):
        block_type = block.get("type")
        title = block.get("title", "")

        html += f"<div class='viz-block'><h3>{title}</h3>"

        if block_type.startswith("chart"):
            html += f"""
            <canvas data-chart='{json.dumps(block["data"])}'></canvas>
            """

        elif block_type == "table":
            headers = block["data"].get("headers", [])
            rows = block["data"].get("rows", [])

            html += "<table><tr>"
            for h in headers:
                html += f"<th>{h}</th>"
            html += "</tr>"

            for row in rows:
                html += "<tr>"
                for cell in row:
                    html += f"<td>{cell}</td>"
                html += "</tr>"

            html += "</table>"

        elif block_type == "timeline":
            html += "<ul>"
            for item in block["data"].get("events", []):
                html += f"<li><b>{item['year']}:</b> {item['event']}</li>"
            html += "</ul>"

        elif block_type == "cards":
            html += "<div class='card-container'>"
            for card in block["data"].get("items", []):
                html += f"""
                <div class='card'>
                    <h4>{card['title']}</h4>
                    <p>{card['value']}</p>
                </div>
                """
            html += "</div>"

        html += "</div>"

    return html

# -------------------------------
# MAIN VISUALIZATION FUNCTION
# -------------------------------
async def visualize_content(prompt_text: str) -> Dict[str, Any]:
    try:
        prompt_text = prompt_text[:2000]

        system_prompt = """
You are a visualization planner.

Convert ANY user prompt into visualization-ready JSON.

Rules:
- Return ONLY valid JSON
- No markdown
- No explanations
- No scripts
- Use mock data if real data is unavailable

Allowed block types:
chart_line, chart_bar, chart_pie, table, timeline, cards

FORMAT:
{
  "title": "Title",
  "blocks": [
    {
      "type": "chart_line",
      "title": "Example",
      "data": {...}
    }
  ]
}
"""

        client = get_visualize_groq_client()

        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3
        )

        plan = json.loads(completion.choices[0].message.content)

        html = build_visualization_html(plan)
        html = sanitize_html(html)

        return {"status": "success", "html": html}

    except Exception as e:
        logger.error("VISUALIZE ERROR", exc_info=True)
        return {
            "status": "failed",
            "html": f"<div><b>Visualization failed:</b> {str(e)}</div>"
        }

