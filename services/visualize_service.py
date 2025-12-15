import json
import asyncio
from typing import Dict, Any
import bleach

from config import logger
from services.ai_service import get_visualize_groq_client


# ============================================================
# HTML SANITIZATION
# ============================================================

ALLOWED_TAGS = [
    "div", "h1", "h2", "h3", "h4", "p",
    "table", "thead", "tbody", "tr", "th", "td",
    "ul", "li", "span", "canvas", "strong", "em"
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


# ============================================================
# HTML BUILDER
# ============================================================

def build_visualization_html(plan: Dict[str, Any]) -> str:
    html = ""

    if "title" in plan:
        html += f"<h2>{plan['title']}</h2>"

    # Intro / explanation text
    for para in plan.get("paragraphs", []):
        html += f"<p>{para}</p>"

    # Visual blocks
    for block in plan.get("blocks", []):
        block_type = block.get("type")
        title = block.get("title", "")

        html += f"<div class='viz-block'>"
        if title:
            html += f"<h3>{title}</h3>"

        if block_type.startswith("chart"):
            html += (
                f"<canvas data-chart='{json.dumps(block['data'])}'></canvas>"
            )

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
                html += f"<li><strong>{item['label']}:</strong> {item['value']}</li>"
            html += "</ul>"

        elif block_type == "cards":
            html += "<div class='card-container'>"
            for card in block["data"].get("items", []):
                html += (
                    "<div class='card'>"
                    f"<h4>{card['title']}</h4>"
                    f"<p>{card['value']}</p>"
                    "</div>"
                )
            html += "</div>"

        html += "</div>"

    return html

def ensure_canvas_exists(html: str) -> str:
    """
    Guarantees at least one canvas exists for frontend rendering.
    """
    if "<canvas" in html:
        return html

    fallback_chart = {
        "type": "bar",
        "labels": ["Phase 1", "Phase 2", "Phase 3"],
        "values": [30, 60, 90]
    }

    html += (
        "<div class='viz-block'>"
        "<h3>Overview</h3>"
        f"<canvas data-chart='{json.dumps(fallback_chart)}'></canvas>"
        "</div>"
    )

    return html



def normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all visualization blocks contain renderable data.
    Backend GUARANTEES frontend-safe visuals.
    """

    normalized_blocks = []

    for block in plan.get("blocks", []):
        btype = block.get("type")
        title = block.get("title", "")
        data = block.get("data") or {}

        # ---------------- TIMELINE ----------------
        if btype == "timeline":
            events = data.get("events")
            if not events:
                data["events"] = [
                    {"label": "Ideation", "value": "Problem discovery and validation"},
                    {"label": "Early Stage", "value": "MVP and first users"},
                    {"label": "Growth", "value": "Scaling revenue and team"},
                ]
            normalized_blocks.append(block)

        # ---------------- CHARTS ----------------
        elif btype and btype.startswith("chart"):
            # enforce chart-ready structure
            labels = data.get("labels")
            values = data.get("values")

            if not labels or not values:
                data["labels"] = ["Stage 1", "Stage 2", "Stage 3"]
                data["values"] = [30, 60, 90]

            # ensure chart type exists
            data.setdefault("type", btype.replace("chart_", ""))

            normalized_blocks.append(block)

        # ---------------- TABLE ----------------
        elif btype == "table":
            if not data.get("rows"):
                data["headers"] = ["Stage", "Focus"]
                data["rows"] = [
                    ["Ideation", "Problem identification"],
                    ["Validation", "Market fit"],
                    ["Growth", "Scaling operations"],
                ]
            normalized_blocks.append(block)

        # ---------------- CARDS ----------------
        elif btype == "cards":
            if not data.get("items"):
                data["items"] = [
                    {"title": "Stage", "value": "Growth"},
                    {"title": "Risk Level", "value": "Medium"},
                ]
            normalized_blocks.append(block)

        # ---------------- UNKNOWN / EMPTY ----------------
        else:
            # drop unusable blocks silently
            continue

    plan["blocks"] = normalized_blocks
    return plan


# ============================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================

async def visualize_content(prompt_text: str) -> Dict[str, Any]:
    prompt_text = prompt_text[:2000]
    client = get_visualize_groq_client()

    system_prompt = """
You are a visualization planner.

Rules:
- Return ONLY valid JSON
- No markdown
- No explanations
- No text outside JSON

Allowed block types:
chart_line, chart_bar, chart_pie, table

JSON FORMAT:
{
  "title": "string",
  "blocks": [
    {
      "type": "chart_line | chart_bar | chart_pie | table",
      "title": "string",
      "data": {
        "labels": [string],
        "values": [number]
      }
    }
  ]
}
"""

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.2
        )

        raw = completion.choices[0].message.content
        logger.info(f"[VISUALIZE RAW JSON]: {raw}")

        plan = json.loads(raw)          # ❗ STRICT
        plan = normalize_plan(plan)     # ❗ SAFE FIXES ONLY

        html = build_visualization_html(plan)
        html = sanitize_html(html)

        return {"status": "success", "html": html}

    except Exception as e:
        logger.error("VISUALIZE FAILED", exc_info=True)
        return {
            "status": "failed",
            "html": "<p>Visualization could not be generated.</p>"
        }
