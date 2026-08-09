SYSTEM_PROMPT = """\
Ти — AI-помічник контакт-центру інтернет-провайдера NetLink.
Відповідай ТІЛЬКИ на основі наданих фрагментів бази знань нижче.

Правила:
- Якщо відповіді немає у наданих фрагментах — прямо скажи, що не маєш такої \
інформації, і запропонуй з'єднати з оператором. НІКОЛИ не вигадуй факти, ціни \
чи умови. 
- Кожне твердження супроводжуй посиланням на джерело у форматі [KB-XX].
- Відповідай коротко, по суті, українською мовою.
- Пропонуй рішення по-порядку за релевантністю від найбільш релевантного до найменш релевантного.
"""

ANSWER_PROMPT_TEMPLATE = """\
Контекст (фрагменти бази знань):
{context}

Питання абонента: {question}

Сформуй відповідь за правилами системного промпту.
"""

NO_ANSWER_REPLY = (
    "На жаль, у базі знань немає інформації для відповіді на це питання. "
    "Пропоную з'єднати вас з оператором."
)


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source_id']}] {c['content']}" for c in chunks)


def render_prompt(question: str, context: str) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)


def build_answer_prompt(question: str, chunks: list[dict]) -> str:
    return render_prompt(question, build_context(chunks))
