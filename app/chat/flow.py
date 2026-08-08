"""Багатокроковий флоу «в абонента не працює інтернет» на LangGraph (Частина B).

Граф:
    START -> collect_info -> [escalate?] -> retrieve_and_answer -> END

collect_info послідовно уточнює 3 слоти (тип проблеми, статус обладнання,
що вже пробували), використовуючи `interrupt()` для паузи й очікування
відповіді абонента. Кожен слот дає до MAX_RETRIES спроб уточнення — якщо
відповідь абонента нерелевантна/порожня, перепитуємо; після вичерпання
спроб — ескалація на оператора.

Коли всі слоти зібрано, retrieve_and_answer підтягує правила з БЗ (RAG),
формує відповідь з посиланнями на джерела і запитує, чи це допомогло.
Негативна відповідь абонента -> ескалація.

Стан персистується через MemorySaver (checkpointer) з thread_id = session_id,
тож flow можна продовжувати між окремими HTTP-запитами до /chat.
"""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.chat.llm_client import llm
from app.chat.prompts import SYSTEM_PROMPT, build_answer_prompt, NO_ANSWER_REPLY
from app.rag.retriever import search, has_relevant_results

MAX_RETRIES = 2

SLOTS = [
    ("problem_type", "Опишіть, будь ласка, у чому саме проблема з інтернетом?"),
    ("equipment_status", "Який колір індикаторів на роутері — «Internet/WAN» та «LOS»?"),
    ("tried_actions", "Що ви вже пробували зробити для вирішення (наприклад, перезавантаження роутера)?"),
]


class FlowState(TypedDict, total=False):
    problem_type: Optional[str]
    equipment_status: Optional[str]
    tried_actions: Optional[str]
    reply: str
    sources: list[str]
    escalate: bool
    resolved: bool


def extract_slot(field: str, question: str, answer: str) -> Optional[str]:
    """LLM-класифікація: чи відповідь абонента містить достатньо інформації для поля."""
    prompt = (
        f"Поле для заповнення: {field}.\n"
        f"Питання, яке ставилось абоненту: {question}\n"
        f"Відповідь абонента: {answer}\n\n"
        "Якщо відповідь містить достатньо інформації для цього поля — поверни "
        "стисле значення поля одним реченням українською. Якщо відповідь порожня, "
        "нерелевантна або ухильна — поверни рівно слово NONE."
    )
    result = llm.invoke([{"role": "user", "content": prompt}]).content.strip()
    if not result or result.upper() == "NONE":
        return None
    return result


def collect_info(state: FlowState) -> dict:
    updates: dict = {}
    for key, question in SLOTS:
        value = state.get(key) or updates.get(key)
        if value:
            continue

        attempts = 0
        while not value:
            prompt = question if attempts == 0 else f"Не зовсім зрозуміло. {question}"
            answer = interrupt(prompt)
            value = extract_slot(key, question, answer)
            if not value:
                attempts += 1
                if attempts > MAX_RETRIES:
                    return {
                        "escalate": True,
                        "reply": "Не вдалося зібрати достатньо інформації. Передаю ваш запит оператору.",
                        "sources": [],
                    }
        updates[key] = value

    return updates


def route_after_collect(state: FlowState) -> str:
    return END if state.get("escalate") else "retrieve_and_answer"


def _classify_yes_no(text: str) -> Optional[bool]:
    lowered = text.lower()
    if any(w in lowered for w in ("так", "да", "yes", "допомогло", "працює")):
        return True
    if any(w in lowered for w in ("ні", "нет", "no", "не допомогло", "не працює")):
        return False
    return None


def retrieve_and_answer(state: FlowState) -> dict:
    query = (
        f"Проблема: {state['problem_type']}. "
        f"Стан обладнання: {state['equipment_status']}. "
        f"Що вже пробували: {state['tried_actions']}."
    )
    results = search(query)

    if not has_relevant_results(results):
        return {"escalate": True, "reply": NO_ANSWER_REPLY, "sources": []}

    prompt = build_answer_prompt(query, results)
    answer = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    ).content
    sources = sorted({r["source_id"] for r in results})

    feedback = interrupt(f"{answer}\n\nЧи це допомогло вирішити проблему? (так/ні)")
    helped = _classify_yes_no(feedback)

    if helped:
        return {"reply": "Радий, що це допомогло! Гарного дня.", "sources": sources, "resolved": True}

    return {
        "escalate": True,
        "reply": "Передаю ваш запит оператору, оскільки запропоноване рішення не допомогло.",
        "sources": sources,
    }


def build_flow_graph():
    graph = StateGraph(FlowState)
    graph.add_node("collect_info", collect_info)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_edge(START, "collect_info")
    graph.add_conditional_edges("collect_info", route_after_collect, ["retrieve_and_answer", END])
    graph.add_edge("retrieve_and_answer", END)
    return graph.compile(checkpointer=MemorySaver())


flow_graph = build_flow_graph()
