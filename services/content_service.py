# stelle_backend/services/content_service.py
import asyncio
from typing import List, Dict, Any, Union, Tuple
from groq import AsyncGroq
import re

from config import logger
from services.ai_service import rate_limited_groq_call, query_internet_via_groq
from services.common_utils import get_current_datetime

async def visual_generate_subqueries(main_query: str, num_subqueries: int = 3) -> list:
    """Generates subqueries specific to visual research."""
    from services.ai_service import client # Use the main shared async client
    
    current_dt = get_current_datetime()
    prompt = (
        f"As of {current_dt} Provide exactly {num_subqueries} distinct search queries related to the following visualization topic:\n"
        f"'{main_query}'\n"
        "Write each query on a separate line, with no numbering or additional text."
    )
    try:
        chat_completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=150,
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content.strip()
        raw_queries = [line.strip() for line in response.splitlines() if line.strip()]
        clean_queries = []
        for q in raw_queries:
            q = re.sub(r"^['\"]|['\"]$", "", q).strip()
            if q:
                clean_queries.append(q)
        unique_subqueries = list(dict.fromkeys(clean_queries))[:num_subqueries]
        logger.info(f"Generated subqueries for '{main_query}' (Visual): {unique_subqueries}")
        return unique_subqueries
    except Exception as e:
        logger.error(f"Visual subquery generation error for '{main_query}': {e}")
        return []

async def visual_synthesize_result(
    main_query: str, contents: list, max_context: int = 4000
) -> str:
    """Synthesizes collected research content for visualization purposes."""
    from services.ai_service import client # Use the main shared async client
    
    trimmed_contents = [c[:1000] for c in contents if c]
    combined_content = " ".join(trimmed_contents)[:max_context]
    prompt = (
        f"Based on the following content, provide a concise, accurate, and well-structured answer to the visualization prompt:\n"
        f"'{main_query}'\n\nContent:\n{combined_content}"
    )
    try:
        chat_completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=500,
            temperature=0.7,
        )
        result = chat_completion.choices[0].message.content.strip()
        logger.info(f"Synthesized result for '{main_query}' (Visual).")
        return result
    except Exception as e:
        logger.error(f"Error during visual synthesis for '{main_query}': {e}")
        return "Error generating the result."


async def generate_html_visualization(content: str) -> str:
    """Generates a complex, interactive HTML document with visualizations."""
    from services.ai_service import client # Use the main shared async client
    
    prompt = f"Generate a complete HTML document code that presents the following research result in a professional and modern layout. The document should have a dark theme with a background color of #000000 and use the brand color #6ee2f5 for accents and interactive elements. Include interactive visualizations such as pie charts, bar graphs, or other diagrams to represent the data mentioned in the result . Use modern CSS styling to create a clean and sleek design, and ensure the layout is structured similar to an A4 sheet, with text explanations and visualizations properly aligned and if you use maintainAspectRatio make sure its true. Incorporate interactive features like hover effects, clickable elements, or animations to enhance user engagement. Output only the HTML code, with all styles and scripts included in the file. Result: '{content}'"
    try:
        completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            max_completion_tokens=8000,
            temperature=0.8,
            reasoning_format="hidden",
        )
        html_code = completion.choices[0].message.content.strip()
        logger.info("Generated HTML visualization code.")
        return html_code
    except Exception as e:
        logger.error(f"Error generating HTML visualization: {e}")
        return "<html><body><h1>Error generating HTML visualization.</h1></body></html>"