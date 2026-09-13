from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from recommendation_engine import recommend


class SongRetriever(BaseRetriever):
    """Wraps the existing cosine-similarity recommender as a LangChain retriever."""

    top_k: int = 7
    filter_mood: bool = True

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        rows = recommend(query, top_k=self.top_k, filter_mood=self.filter_mood)
        return [
            Document(
                page_content=f"{r.title} by {r.artist} ({r.year}), mood: {r.mood}",
                metadata={
                    "title": r.title,
                    "artist": r.artist,
                    "year": r.year,
                    "mood": r.mood,
                    "score": round(float(r.scores), 3),
                },
            )
            for r in rows.itertuples()
        ]


_SYSTEM_TEMPLATE = """You are a music recommender. Below are songs from our catalog that are most relevant to the user's request. Recommend them in 2-3 friendly sentences, briefly saying why each fits. Do not invent songs that are not listed.

Relevant songs:
{context}"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_TEMPLATE),
        ("human", "{input}"),
    ]
)
_llm = ChatOllama(model="llama3.2:3b", temperature=0)


def ask(query, top_k=7, filter_mood=True):
    """Run the RAG pipeline: retrieve songs, augment the prompt, generate an answer.

    Returns (answer: str, songs: list[dict]).
    """
    # R — retrieve using the existing cosine-similarity engine.
    retriever = SongRetriever(top_k=top_k, filter_mood=filter_mood)
    docs = retriever.invoke(query)

    # A — augment: inject the retrieved songs into the prompt as context.
    context = "\n".join(d.page_content for d in docs)
    messages = _prompt.format_messages(input=query, context=context)

    # G — generate a natural-language answer.
    answer = _llm.invoke(messages).content

    return answer, [d.metadata for d in docs]
