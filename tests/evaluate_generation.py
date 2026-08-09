import json
import os
import logging

from openai import OpenAI

from app.rag.retriever import search, has_relevant_results
from app.chat.prompts import build_context, NO_ANSWER_REPLY
from app.chat.service import generate_answer_text


logging.basicConfig(level=logging.INFO)

DATASET_PATH = "tests/golden_dataset.json"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

client = OpenAI()


JUDGE_PROMPT = """
Ти — evaluator для RAG-системи підтримки інтернет-провайдера.

Оціни відповідь агента за двома критеріями.

1. FAITHFULNESS
Чи містить відповідь тільки інформацію, яку можна підтвердити
наданим контекстом?

2. CORRECTNESS
Чи правильно відповідь відповідає на питання користувача
та очікуваній відповіді?

Шкала для кожного критерію:

0 = неправильно / суперечить контексту
1 = частково правильно
2 = повністю правильно

ВАЖЛИВО:
- Не оцінюй стиль відповіді.
- Не додавай власних знань.
- Якщо expected_sources порожній, правильна поведінка —
  не вигадувати відповідь, а повідомити, що інформації в базі знань немає.
- Якщо в answer є інформація, якої немає в context,
  це знижує faithfulness.

Поверни ТІЛЬКИ JSON такого формату:

{
  "faithfulness": 0,
  "correctness": 0,
  "reason": "коротке пояснення"
}
"""


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def judge_answer(
    question: str,
    expected_answer: str,
    context: str,
    answer: str,
    expected_sources: list[str],
):
    prompt = f"""
{JUDGE_PROMPT}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

EXPECTED SOURCES:
{expected_sources}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return json.loads(response.choices[0].message.content)


def evaluate_case(case):
    question = case["question"]
    expected_answer = case["expected_answer"]
    expected_sources = case.get("expected_sources", [])

    print("\n" + "-" * 80)
    print(f"Question: {question}")

    # 1. Retrieval — той самий шлях, що й у продакшн-коді (app/chat/service.py)
    results = search(question, limit=3)

    # 2. Generation — якщо релевантних чанків немає, повторюємо поведінку
    #    продакшену: фіксована відповідь без виклику LLM, без вигаданих фактів.
    if not has_relevant_results(results):
        context = ""
        answer = NO_ANSWER_REPLY
    else:
        context = build_context(results)
        answer = generate_answer_text(question, context)

    print(f"\nExpected answer:\n{expected_answer}")
    print(f"\nGenerated answer:\n{answer}")

    # 3. LLM-as-a-judge
    evaluation = judge_answer(
        question=question,
        expected_answer=expected_answer,
        context=context,
        answer=answer,
        expected_sources=expected_sources,
    )

    faithfulness = evaluation["faithfulness"]
    correctness = evaluation["correctness"]

    print(f"\nFaithfulness: {faithfulness}/2")
    print(f"Correctness:  {correctness}/2")
    print(f"Reason:       {evaluation.get('reason', '')}")

    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness,
        "correctness": correctness,
        "reason": evaluation.get("reason", ""),
    }


def main():
    dataset = load_dataset()

    results = []

    total_faithfulness = 0
    total_correctness = 0

    for case in dataset:
        try:
            result = evaluate_case(case)

            results.append(result)

            total_faithfulness += result["faithfulness"]
            total_correctness += result["correctness"]

        except Exception:
            logging.exception(
                "Generation evaluation failed for question: %s",
                case["question"],
            )

    if not results:
        print("No evaluation results.")
        return

    count = len(results)

    # 0..2 -> 0..100%
    faithfulness_percent = (
        total_faithfulness / (count * 2)
    ) * 100

    correctness_percent = (
        total_correctness / (count * 2)
    ) * 100

    print("\n")
    print("=" * 50)
    print("GENERATION")
    print("=" * 50)

    print(
        f"faithfulness: {faithfulness_percent:.2f}%"
    )

    print(
        f"correctness:  {correctness_percent:.2f}%"
    )

    print(
        f"cases:        {count}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
