"""Основна логіка чату: маршрутизація між LangGraph-флоу та вільним RAG Q&A."""

from langgraph.types import Command

from app.chat.flow import flow_graph, extract_slot, SLOTS
from app.chat.llm_client import llm
from app.chat.prompts import SYSTEM_PROMPT, build_answer_prompt, NO_ANSWER_REPLY
from app.rag.retriever import search, has_relevant_results

_TROUBLESHOOT_TRIGGERS = ("не працює інтернет", "немає інтернету", "пропав інтернет")

# session_id -> активний thread флоу; для продакшн потрібне зовнішнє сховище (Redis тощо).
_active_flows: set[str] = set()


def _looks_like_troubleshoot_request(message: str) -> bool:
    lowered = message.lower()
    return any(trigger in lowered for trigger in _TROUBLESHOOT_TRIGGERS)


def generate_answer(question: str) -> dict:
    results = search(question)

    if not has_relevant_results(results):
        return {"reply": NO_ANSWER_REPLY, "sources": [], "escalate": False}

    prompt = build_answer_prompt(question, results)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return {
        "reply": response.content,
        "sources": sorted({r["source_id"] for r in results}),
        "escalate": False,
    }


def _run_flow(session_id: str, resume_value: str | None, initial_state: dict | None = None) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    graph_input = initial_state if initial_state is not None else Command(resume=resume_value)
    result = flow_graph.invoke(graph_input, config=config)

    if "__interrupt__" in result:
        _active_flows.add(session_id)
        question = result["__interrupt__"][0].value
        return {"reply": question, "sources": [], "escalate": False}

    _active_flows.discard(session_id)
    return {
        "reply": result.get("reply", ""),
        "sources": result.get("sources", []),
        "escalate": result.get("escalate", False),
    }


def handle_message(session_id: str, message: str) -> dict:
    if not message.strip():
        raise ValueError("message is required")

    if session_id in _active_flows:
        return _run_flow(session_id, resume_value=message)

    if _looks_like_troubleshoot_request(message):
        problem_key, problem_question = SLOTS[0]
        seeded = extract_slot(problem_key, problem_question, message)
        initial_state = {problem_key: seeded} if seeded else {}
        return _run_flow(session_id, resume_value=None, initial_state=initial_state)

    return generate_answer(message)
