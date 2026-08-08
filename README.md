# NetLink AI Assistant

AI-помічник контакт-центру інтернет-провайдера NetLink: RAG над базою знань +
багатокроковий флоу діагностики «не працює інтернет» на LangGraph.

## Структура проєкту

```
app/
├── api/
│   ├── documents.py   # POST /documents/upload — індексація файлу в базу знань
│   └── chat.py        # POST /chat — основний ендпоінт діалогу
├── rag/
│   ├── chunker.py      # витяг тексту з файлу + розбиття на чанки
│   ├── embeddings.py   # обгортка над OpenAI Embeddings
│   └── retriever.py    # векторний пошук по document_chunks + індексація документа
├── chat/
│   ├── service.py      # маршрутизація: флоу-сценарій vs вільний RAG Q&A
│   ├── prompts.py       # системний промпт генерації (цитування, заборона вигадувати)
│   ├── flow.py          # LangGraph-флоу «не працює інтернет»
│   └── llm_client.py    # спільний ChatOpenAI клієнт
├── db/
│   ├── database.py      # пул з'єднань PostgreSQL + pgvector
│   └── models.py        # dataclasses Document / DocumentChunk
└── main.py              # FastAPI app, реєстрація роутерів

db/
└── init.sql             # схема БД (documents, document_chunks), застосовується автоматично

docker-compose.yml        # PostgreSQL + pgvector в контейнері
```

## Вимоги

- Python 3.11+
- Docker + Docker Compose (найпростіший спосіб підняти БД) — або власний
  PostgreSQL з розширенням [pgvector](https://github.com/pgvector/pgvector)
- OpenAI API key

## Встановлення

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install fastapi uvicorn[standard] psycopg2-binary pgvector \
    langchain-openai langchain-text-splitters langgraph \
    python-docx python-dotenv
```

## Налаштування `.env`

Створіть `.env` у корені проєкту:

```
OPENAI_API_KEY=sk-...
DB_NAME=netlink
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
```

## Підготовка бази даних

### Варіант 1 — Docker (рекомендовано)

Піднімає PostgreSQL з розширенням pgvector і одразу застосовує схему з
`db/init.sql` (таблиці `documents`, `document_chunks` створюються автоматично
при першому старті контейнера):

```bash
docker compose up -d
```

Значення `DB_NAME` / `DB_USER` / `DB_PASSWORD` / `DB_PORT` беруться з вашого
`.env` (див. розділ нижче); якщо `.env` ще не створено — контейнер підніметься
зі значеннями за замовчуванням (`netlink` / `postgres` / `postgres` / `5432`).

Перевірити, що контейнер живий:

```bash
docker compose ps
docker compose logs -f db
```

Зупинити: `docker compose down` (дані залишаються у volume `db_data`).
Повністю скинути БД: `docker compose down -v`.

> Якщо порт 5432 вже зайнятий іншим контейнером/локальним PostgreSQL —
> змініть `DB_PORT` у `.env` (наприклад, на `5433`) і перезапустіть
> `docker compose up -d`; docker-compose.yml прокидає порт з `.env` автоматично.
> Якщо у вас вже є контейнер з pgvector, який використовує проєкт (наприклад
> `postgres-local`), можна не піднімати новий — просто перевірте, що в ньому
> застосована схема з `db/init.sql`, і вкажіть його host/port/креденшели в `.env`.

### Варіант 2 — власний PostgreSQL

Якщо БД вже є, застосуйте схему вручну:

```bash
psql -h <host> -U <user> -d <db> -f db/init.sql
```

> Розмірність вектора (1536) відповідає моделі `text-embedding-ada-002` /
> `text-embedding-3-small` за замовчуванням у `OpenAIEmbeddings`. Якщо оберете
> іншу модель — оновіть розмірність у `db/init.sql`.

## Запуск

```bash
uvicorn app.main:app --reload
```

Сервер підніметься на `http://127.0.0.1:8000`, інтерактивна документація —
`http://127.0.0.1:8000/docs`.

## Використання

### Завантаження документа в базу знань

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@kb-01.txt"
```

Підтримувані формати: `.txt`, `.md`, `.docx`. Ліміт розміру файлу — 10 МБ.

### Чат

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "1", "message": "Скільки коштує тариф Гіга 1000?"}'
```

Відповідь:

```json
{
  "reply": "...",
  "sources": ["KB-11"],
  "escalate": false
}
```

Якщо повідомлення схоже на скаргу «не працює інтернет», сервіс запускає
LangGraph-флоу: послідовно уточнює тип проблеми, стан обладнання й що вже
пробували (до 2 повторних уточнень на крок, інакше — ескалація), після чого
підтягує релевантні статті з бази знань і формує відповідь із проханням
підтвердити, чи це допомогло. Продовжуйте діалог, надсилаючи наступні
повідомлення з тим самим `session_id`.

## Retrieval — підхід і обмеження

- **Chunking**: `RecursiveCharacterTextSplitter`, розмір чанка 300 символів,
  overlap 100 — підібрано під короткі статті бази знань NetLink (один пункт
  KB зазвичай вкладається в чанк цілком, overlap не рве факт на межі).
- **Індексація**: OpenAI embeddings + pgvector, пошук за cosine distance
  (`<=>`), `ivfflat`-індекс.
- **Відсічка "немає відповіді"**: якщо відстань до найближчого чанка вища за
  поріг (`NO_ANSWER_DISTANCE_THRESHOLD` у `app/rag/retriever.py`), відповідь
  не генерується — користувачу пропонується оператор.

**Коли підхід спрацює погано:**
- Питання, що перефразовує факт словами, відсутніми в базі (chunk embedding
  далекий за косинусом, хоча тема та сама) — dense retrieval без BM25/гібридного
  пошуку може не знайти релевантний чанк.
- Питання, що потребують агрегації кількох статей (наприклад, порівняння
  двох тарифів) — top-k чанки можуть не покрити обидва джерела одночасно.
- Поріг відсічки — фіксоване число, підібране евристично; на реальних даних
  його потрібно калібрувати на golden dataset (Частина C).

**Що зробити далі:** гібридний пошук (dense + BM25/keyword), re-ranking
top-k результатів, розширення запиту (query expansion) синонімами продукту,
калібрування порогу на розміченому наборі питань.
