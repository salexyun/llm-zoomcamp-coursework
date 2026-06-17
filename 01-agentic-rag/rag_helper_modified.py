from dataclasses import dataclass
from textwrap import dedent
from typing import Any

INSTRUCTIONS = dedent("""
    You are a helpful teaching assistant for the LLM Zoomcamp.

    Answer questions using only the provided lesson content.

    If the answer cannot be found in the provided context,
    reply with "I don't know."
""").strip()

PROMPT_TEMPLATE = dedent("""
    QUESTION: {question}

    CONTEXT: {context}
""").strip()


@dataclass
class RAGResult:
    """Result returned by a RAG query."""

    output_text: str
    input_tokens: int


class RAGBase:
    """Simple retrieval-augmented generation helper."""

    def __init__(
        self,
        index: Any,
        llm_client: Any,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-5.4-mini",
    ) -> None:
        """Initialize the RAG helper with a search index and LLM client."""
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query: str, num_results: int = 5):
        """Search the index for document relevant to the query."""
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results: str):
        '"""Biuld a text context from retrieved documents.'
        lines = []

        for doc in search_results:
            lines.append(doc["filename"])
            lines.append("Content: " + doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: str):
        """Build a prompt from the query and retrieved documents."""
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str):
        """Send a prompt to the LLM and return the full response."""
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        return self.llm_client.responses.create(model=self.model, input=input_messages)

    def rag(self, query: str):
        """Answer a query using retrieval-augmented generation."""
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)

        return RAGResult(
            output_text=response.output_text, input_tokens=response.usage.input_tokens
        )
