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

# ✅ ADDED data-heatmap ONLY
ALLOWED_ATTRS = {
    "*": ["class", "data-chart", "data-heatmap"]
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

        # ---------------- CHARTS ----------------
        if block_type.startswith("chart"):
            html += (
                f"<canvas data-chart='{json.dumps(block['data'])}'></canvas>"
            )

        # ✅ ADDED HEATMAP (ONLY THIS BLOCK)
        elif block_type == "heatmap":
            html += (
                f"<canvas data-heatmap='{json.dumps(block['data'])}'></canvas>"
            )

        # ---------------- TABLE ----------------
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

        # ---------------- TIMELINE ----------------
        elif block_type == "timeline":
            html += "<ul>"
            for item in block["data"].get("events", []):
                html += f"<li><strong>{item['label']}:</strong> {item['value']}</li>"
            html += "</ul>"

        # ---------------- CARDS ----------------
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


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    normalized_blocks = []

    for block in plan.get("blocks", []):
        btype = block.get("type")
        data = block.get("data") or {}

        # ---------------- CHARTS ----------------
        if btype and btype.startswith("chart"):
            labels = data.get("labels")
            values = data.get("values")

            if not labels or not values:
                data["labels"] = ["Stage 1", "Stage 2", "Stage 3"]
                data["values"] = [30, 60, 90]

            # ✅ FIX: legend label for frontend (prevents "undefined")
            data.setdefault("label", block.get("title", "Value"))

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

        # ---------------- HEATMAP ----------------
        elif btype == "heatmap":
            if (
                not data.get("xLabels")
                or not data.get("yLabels")
                or not data.get("values")
            ):
                data["xLabels"] = ["A", "B", "C"]
                data["yLabels"] = ["X", "Y"]
                data["values"] = [
                    [1, 2, 3],
                    [4, 5, 6]
                ]
            normalized_blocks.append(block)

        # ---------------- UNKNOWN ----------------
        else:
            continue

    plan["blocks"] = normalized_blocks
    return plan


# ============================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================

async def visualize_content(prompt_text: str) -> Dict[str, Any]:
    prompt_text = prompt_text[:2000]
    client = get_visualize_groq_client()

    # ✅ ONLY CHANGE: added heatmap to allowed types
    system_prompt = """
You are a visualization planner.

Rules:
- Return ONLY valid JSON
- No markdown
- No explanations
- No text outside JSON

Allowed block types:
chart_line, chart_bar, chart_pie, table, heatmap

JSON FORMAT:
{
  "title": "string",
  "blocks": [
    {
      "type": "chart_line | chart_bar | chart_pie | table | heatmap",
      "title": "string",
      "data": {}
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

        plan = json.loads(raw)
        plan = normalize_plan(plan)

        html = build_visualization_html(plan)
        html = sanitize_html(html)

        return {"status": "success", "html": html}

    except Exception:
        logger.error("VISUALIZE FAILED", exc_info=True)
        return {
            "status": "failed",
            "html": "<p>Visualization could not be generated.</p>"
        }
