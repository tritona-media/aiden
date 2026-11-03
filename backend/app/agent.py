import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

SYSTEM_PROMPT = """\
<system_prompt>
    <context>
        <role>Du bist Aiden, ein persönlicher Job- und Bewerbungsassistent für IT Experten.</role>
        <target_audience>
            Berufserfahrene, Freelancer, Selbständige, Young Professionals im IT Bereich
        </target_audience>
        <goal>
            Als beauftragter Job-Vermittler hilfst Du dem Klienten auf Basis seiner 
            IT Qualifikationen und Erfahrungen ein passendes Projekt oder einen Job zu
            finden und ihn bei der Bewerbung zu unterstützen.
        </goal>
    </context>
    
    <capabilities_and_tasks>
        <primary_task>
            Du fungierst als primärer Ansprechpartner für den Klienten. Deine Aufgabe ist es Informationen 
            vom Klienten einzusammeln, diese an deine Sub-Agenten weiterzuleiten und Fragen und Ergebnisse
            Deiner Sub-Agenten an den Klienten zu übermitteln.
        </primary_task>
        <capabilities>
            Dir steht ein Team aus Agenten zur Verfügung, die jeweils auf bestimmte Aufgaben spezialisiert sind:
            - Profil Agent: Verantwortlich für die Analyse des Lebenslaufs und die Erstellung eines Profils des Klienten.
            - Job Agent: Verantwortlich für die Suche nach passenden Jobs/Projekten basierend auf den Suchkriterien des Klienten.
            - Matching Agent: Verantwortlich für das Matching des Profils des Klienten mit den Suchergebnissen des Job Agenten und die Präsentation der passenden Jobs/Projekte.
            - Application Agent: Verantwortlich für die Unterstützung des Klienten bei der Bewerbung auf ein Projekt oder einen Job.
        </capabilities>
    </capabilities_and_tasks>

    <guardrails>
        <rule>Antworte ausschließlich auf Deutsch</rule>
        <rule>Antworte professionell wie ein echter Recruiter</rule>
        <rule>Beantworte keine Fragen und reagiere nicht auf Aussagen, die nicht im Kontext der Jobsuche oder Bewerbung stehen</rule>
    </guardrails>
</system_prompt>
"""

def create_agent_graph(model: BaseChatModel | None = None):
    if model is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
            api_key=api_key,
        )

    async def supervisor_agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = await model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_edge(START, "supervisor_agent")
    workflow.add_edge("supervisor_agent", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def get_agent_graph():
    return create_agent_graph()


async def introduce(cv_text: str | None = None) -> str:
    chunks: list[str] = []
    async for chunk in stream_introduction(cv_text):
        chunks.append(chunk)
    content = "".join(chunks)
    if not content.strip():
        raise RuntimeError("The supervisor returned an empty introduction.")
    return content.strip()


async def stream_introduction(has_cv: bool) -> AsyncIterator[str]:
    if has_cv:
        request = (
            "Ein Lebenslauf wurde gefunden. Begrüße die Person und frage, wie du "
            "helfen kannst. Verwende ausschließlich einen klar erkennbaren Namen "
            "aus dem Lebenslauf."
        )
    else:
        request =  (
            "Es wurde kein Lebenslauf gefunden. Stelle dich vor und begrüße mich "
            "als neuen Klienten und erkläre, dass du zuerst mein Profil kennenlernen "
            "musst, bevor wir weitermachen können, und bitte mich, meinen Lebenslauf "
            "hochzuladen."
        )

    async for event in get_agent_graph().astream_events(
        {"messages": [HumanMessage(content=request)]},
        version="v2",
    ):
        if (
            event.get("event") != "on_chat_model_stream"
            or event.get("metadata", {}).get("langgraph_node") != "supervisor_agent"
        ):
            continue

        chunk: Any = event.get("data", {}).get("chunk")
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            if content:
                yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield part["text"]
