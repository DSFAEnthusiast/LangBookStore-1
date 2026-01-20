"""
RightBookAI (initial agent)

Current behavior:
- Routes the user's question directly to the LLM (no tools / no retrieval yet).

Prereqs:
- Set OPENAI_API_KEY (supports loading from .env.local / .env)
- Optional: set RIGHTBOOKAI_MODEL to override the model name
- Install deps:
  pip install -U "langchain[openai]"
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Final

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

try:
    from openai import NotFoundError as OpenAINotFoundError
except Exception:  # pragma: no cover
    OpenAINotFoundError = None  # type: ignore[assignment]

from tools.budget_bundler import budget_bundler
from tools.get_answers import get_answers
from tools.recommend_books import recommend_books


SYSTEM_PROMPT: Final[str] = (
    "You are RightBookAI, a proper British concierge for LangBookstore. "
    "Be polished, warm, and precise. "
    "You help customers navigate our bookstore by answering questions about available books, "
    "making tailored recommendations, and assembling suggested orders within a stated budget and interests. "
    "Ask brief clarifying questions when needed, and present options with clear reasoning and prices when relevant."
)
MODEL_NAME: Final[str] = "gpt-5-nano"


def choose_tool_name(user_query: str) -> str:
    """
    Heuristic router (placeholder).

    Returns one of: "GetAnswers" | "RecommendBooks" | "BudgetBundler"
    """
    q = user_query.lower()
    if any(k in q for k in ["budget", "$", "under ", "within ", "spend", "dollars"]):
        return "BudgetBundler"
    if any(
        k in q
        for k in [
            "recommend",
            "suggest",
            "what should i read",
            "similar to",
            "next read",
        ]
    ):
        return "RecommendBooks"
    return "GetAnswers"


def rightbookai_route_to_tool_placeholder(user_query: str) -> str:
    """Return the placeholder response for whichever tool would be used."""
    _, result = rightbookai_route_to_tool_placeholder_with_meta(user_query)
    return result


def rightbookai_route_to_tool_placeholder_with_meta(user_query: str) -> tuple[str, str]:
    """Return (tool_name, result) for the placeholder tool router."""
    tool_name = choose_tool_name(user_query)
    if tool_name == "BudgetBundler":
        return tool_name, budget_bundler.invoke({"budget_request": user_query})
    if tool_name == "RecommendBooks":
        return tool_name, recommend_books.invoke({"user_request": user_query})
    return tool_name, get_answers.invoke({"query": user_query})

def _load_env() -> None:
    """
    Load env vars from local dotenv files.

    Priority:
    - existing process env
    - .env.local
    - .env
    """
    # Don't override already-set environment variables.
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


def rightbookai_answer(question: str) -> str:
    """Answer a user question (placeholder tool routing only)."""
    return rightbookai_route_to_tool_placeholder(question)


def rightbookai_answer_via_llm(question: str) -> str:
    """Route a user question directly to the LLM (no tools / no retrieval)."""
    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Set it in the environment or in .env.local, then re-run."
        )
    model_name = os.environ.get("RIGHTBOOKAI_MODEL", MODEL_NAME)
    model = init_chat_model(model=model_name, model_provider="openai")
    try:
        ai_msg = model.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", question),
            ]
        )
    except Exception as e:
        # Provide a clearer message for the common "model not found / no access" case.
        if OpenAINotFoundError is not None and isinstance(e, OpenAINotFoundError):
            raise SystemExit(
                f"Model '{model_name}' was not found or your API key doesn't have access.\n"
                f"Set RIGHTBOOKAI_MODEL in .env.local to a model you have access to (e.g. gpt-4.1-mini), then re-run."
            ) from e
        raise
    return ai_msg.text


def _read_question_from_cli(argv: list[str]) -> str:
    if len(argv) > 1:
        return " ".join(argv[1:]).strip()
    # If running interactively, don't block on sys.stdin.read() (which waits for EOF).
    if sys.stdin.isatty():
        return input("Question: ").strip()
    return sys.stdin.read().strip()


def _init_cli_colors() -> tuple[bool, object]:
    """
    Initialize cross-platform ANSI color support (Windows-friendly).

    Returns: (enabled, color_module_or_None)
    """
    # Avoid emitting ANSI escape codes into piped output / logs.
    if not sys.stdout.isatty():
        return False, None
    if os.environ.get("NO_COLOR"):
        return False, None
    try:
        import colorama  # type: ignore

        # Enables ANSI colors in Windows terminals; no-op elsewhere.
        colorama.just_fix_windows_console()
        return True, colorama
    except Exception:
        return False, None


def _format_cli_output(*, question: str, tool_name: str, response: str) -> str:
    colors_enabled, colorama = _init_cli_colors()

    if colors_enabled and colorama is not None:
        Fore = colorama.Fore
        Style = colorama.Style

        def c(s: str, *, fore: str = "", bright: bool = False) -> str:
            return f"{Style.BRIGHT if bright else ''}{fore}{s}{Style.RESET_ALL}"

        title = c("RightBookAI", fore=Fore.CYAN, bright=True)
        q_hdr = c("Question", fore=Fore.YELLOW, bright=True)
        r_hdr = c("Routed to", fore=Fore.MAGENTA, bright=True)
        a_hdr = c("Response", fore=Fore.GREEN, bright=True)
    else:
        title = "RightBookAI"
        q_hdr = "Question"
        r_hdr = "Routed to"
        a_hdr = "Response"

    sep = "-" * 72
    q_body = textwrap.indent(question.strip(), "  ") if question.strip() else "  (empty)"
    r_body = f"  {tool_name}"
    a_body = textwrap.indent(response.rstrip(), "  ") if response.strip() else "  (no response)"

    return "\n".join(
        [
            sep,
            title,
            sep,
            "",
            f"{q_hdr}:",
            q_body,
            "",
            f"{r_hdr}:",
            r_body,
            "",
            f"{a_hdr}:",
            a_body,
            "",
            sep,
        ]
    )


if __name__ == "__main__":
    # Interactive mode (no args, stdin is a TTY): simple REPL loop.
    if len(sys.argv) == 1 and sys.stdin.isatty():
        while True:
            question = _read_question_from_cli(sys.argv)
            if not question or question.lower() in {"exit", "quit"}:
                raise SystemExit(0)
            tool_name, result = rightbookai_route_to_tool_placeholder_with_meta(question)
            print(_format_cli_output(question=question, tool_name=tool_name, response=result))
        raise SystemExit(0)

    # Non-interactive / one-shot mode: args or piped stdin.
    question = _read_question_from_cli(sys.argv)
    if not question:
        raise SystemExit("Provide a question as args or via stdin.")
    tool_name, result = rightbookai_route_to_tool_placeholder_with_meta(question)
    print(_format_cli_output(question=question, tool_name=tool_name, response=result))

