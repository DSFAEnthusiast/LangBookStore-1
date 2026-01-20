"""
RecommendBooks Tool (placeholder)

Purpose:
- Take in user preferences to recommend books to try next.

Note:
- Placeholder only. Does NOT yet compute real recommendations from `storedata.json`.
"""

from __future__ import annotations

from langchain.tools import tool


@tool
def recommend_books(user_request: str) -> str:
    """Recommend books based on user preferences (placeholder)."""
    return f"[TOOL: RecommendBooks] I would use RecommendBooks to handle: {user_request}"

