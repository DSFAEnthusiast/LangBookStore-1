"""
BudgetBundler Tool (placeholder)

Purpose:
- Take in a budget and interests and suggest a bundle of books to purchase.

Note:
- Placeholder only. Does NOT yet optimize against prices/sale prices in `storedata.json`.
"""

from __future__ import annotations

from langchain.tools import tool


@tool
def budget_bundler(budget_request: str) -> str:
    """Assemble a suggested book order within a stated budget (placeholder)."""
    return f"[TOOL: BudgetBundler] I would use BudgetBundler to handle: {budget_request}"

