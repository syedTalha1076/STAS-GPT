import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# IMPORTS
# ============================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.messages import SystemMessage

from langgraph.graph import (
    StateGraph,
    START,
    MessagesState,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools


# ============================================================
# DIRECTORIES
# ============================================================

Path("data").mkdir(exist_ok=True)


# ============================================================
# GEMINI MODELS
# ============================================================

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)


ALLOWED_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
}


# ============================================================
# GROQ FALLBACK MODEL
# ============================================================

# This model is used automatically when Gemini
# returns a quota/rate-limit error.

GROQ_FALLBACK_MODEL = os.getenv(
    "GROQ_FALLBACK_MODEL",
    "llama-3.3-70b-versatile",
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named STAS GPT, similar to ChatGPT.

You can:

1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.
8. Use weather for weather updates.
9. Send an email when the user asks for email sending.
10. Use stock tools when the user asks about stocks or financial market data.
11. Use create document tool when user asks for creating documents.

Rules:

- If the user asks about latest news, current events, recent updates,
  today's information, current prices, current people, current versions,
  new releases, or anything time-sensitive, use Tavily Search.

- If the user asks about an uploaded document, use
  search_uploaded_documents.

- If the user asks you to remember something, use remember_this.

- If the user asks about previous preferences or saved facts,
  use recall_memory.

- Use calculator for math questions.

- If the user asks for current weather information,
  use the weather tool.

- If the user asks to send an email, the user should provide
  the recipient email address. Create a suitable subject and body
  and send the email using the email tool.

- If the user asks about stocks, stock prices, market data,
  or financial information, use the appropriate stock tool.

- When using web search, summarize the information clearly.

- If the user asks to create a PDF or DOCX document,
  use the create_document tool.

- Be clear, helpful, and concise.


OUTPUT FORMATTING RULES:

- Always respond using clean plain text.
- Do NOT use Markdown syntax.
- Never use ###, ##, # for headings.
- Never use ** for bold text.
- Never use * for italic text.
- Never use backticks for code or emphasis unless the user explicitly asks for Markdown.
- Do not put Markdown links in your normal response.
- Use simple headings without # symbols.
- Use normal numbered lists such as:
  1. First item
  2. Second item
  3. Third item
- Use simple bullet points when appropriate.
- Keep responses clean, professional, and easy to read.
- When creating a PDF or DOCX, the document content must also contain no Markdown syntax.
"""


# ============================================================
# MODEL VALIDATION
# ============================================================

def normalize_model_name(model_name: str | None) -> str:
    """
    Validate the selected Gemini model from the frontend.

    If the model is missing or unavailable,
    use DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# ============================================================
# CHECK IF ERROR IS GEMINI QUOTA/RATE LIMIT ERROR
# ============================================================

def is_gemini_quota_error(error: Exception) -> bool:
    """
    Detect Gemini quota/rate-limit errors.

    Examples include:

    429
    RESOURCE_EXHAUSTED
    quota exceeded
    rate limit
    generate_content_free_tier
    """

    error_text = str(error).lower()

    quota_indicators = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "quotaexceeded",
        "rate limit",
        "ratelimit",
        "generate_content_free_tier",
        "generaterequestsperday",
        "resource exhausted",
    ]

    return any(
        indicator in error_text
        for indicator in quota_indicators
    )


# ============================================================
# BUILD AGENT
# ============================================================

def build_agent(model_name: str):

    selected_model = normalize_model_name(
        model_name
    )

    print()
    print("=" * 60)
    print("STAS GPT MODEL CONFIGURATION")
    print("=" * 60)

    print(
        f"Primary Gemini model : {selected_model}"
    )

    print(
        f"Fallback Groq model  : {GROQ_FALLBACK_MODEL}"
    )

    print("=" * 60)
    print()


    # ========================================================
    # PRIMARY GEMINI LLM
    # ========================================================

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        streaming=True,
    )


    # ========================================================
    # FALLBACK GROQ LLM
    # ========================================================

    fallback_llm = ChatGroq(
        model=GROQ_FALLBACK_MODEL,
        streaming=True,
    )


    # ========================================================
    # BIND TOOLS
    # ========================================================

    llm_with_tools = llm.bind_tools(
        tools
    )

    fallback_llm_with_tools = fallback_llm.bind_tools(
        tools
    )


    # ========================================================
    # CHATBOT NODE
    # ========================================================

    def chatbot_node(
        state: MessagesState
    ):

        messages = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ] + state["messages"]


        # ====================================================
        # TRY PRIMARY GEMINI MODEL
        # ====================================================

        try:

            print(
                f"Using Gemini: {selected_model}"
            )

            response = (
                llm_with_tools.invoke(
                    messages
                )
            )

            return {
                "messages": [
                    response
                ]
            }


        # ====================================================
        # GEMINI ERROR
        # ====================================================

        except Exception as gemini_error:

            print()
            print(
                "Gemini error:"
            )

            print(
                gemini_error
            )

            print()


            # =================================================
            # CHECK WHETHER IT IS A QUOTA ERROR
            # =================================================

            if is_gemini_quota_error(
                gemini_error
            ):

                print(
                    "Gemini quota/rate limit detected."
                )

                print(
                    "Switching to Groq fallback..."
                )

                print(
                    f"Groq model: {GROQ_FALLBACK_MODEL}"
                )


                # =============================================
                # TRY GROQ FALLBACK
                # =============================================

                try:

                    response = (
                        fallback_llm_with_tools.invoke(
                            messages
                        )
                    )

                    print(
                        "Groq fallback successful."
                    )

                    return {
                        "messages": [
                            response
                        ]
                    }


                # =============================================
                # GROQ ALSO FAILED
                # =============================================

                except Exception as groq_error:

                    print()
                    print(
                        "Groq fallback error:"
                    )

                    print(
                        groq_error
                    )

                    print()

                    # Return the original Gemini error
                    # because Gemini was the primary model.
                    raise RuntimeError(
                        "Both Gemini and Groq failed. "
                        f"Gemini error: {gemini_error} | "
                        f"Groq error: {groq_error}"
                    )


            # =================================================
            # NON-QUOTA GEMINI ERROR
            # =================================================

            raise


    # ========================================================
    # TOOL NODE
    # ========================================================

    tool_node = ToolNode(
        tools
    )


    # ========================================================
    # LANGGRAPH WORKFLOW
    # ========================================================

    workflow = StateGraph(
        MessagesState
    )


    # ========================================================
    # ADD NODES
    # ========================================================

    workflow.add_node(
        "chatbot",
        chatbot_node,
    )

    workflow.add_node(
        "tools",
        tool_node,
    )


    # ========================================================
    # START → CHATBOT
    # ========================================================

    workflow.add_edge(
        START,
        "chatbot",
    )


    # ========================================================
    # CHATBOT → TOOLS OR END
    # ========================================================

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )


    # ========================================================
    # TOOLS → CHATBOT
    # ========================================================

    workflow.add_edge(
        "tools",
        "chatbot",
    )


    # ========================================================
    # SQLITE CHECKPOINT
    # ========================================================

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False,
    )


    checkpointer = SqliteSaver(
        conn
    )


    # ========================================================
    # COMPILE WORKFLOW
    # ========================================================

    return workflow.compile(
        checkpointer=checkpointer
    )


# ============================================================
# AGENT CACHE
# ============================================================

_AGENT_CACHE = {}


# ============================================================
# GET AGENT
# ============================================================

def get_agent(
    model_name: str | None = None
):

    selected_model = normalize_model_name(
        model_name
    )


    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[
            selected_model
        ] = build_agent(
            selected_model
        )


    return _AGENT_CACHE[
        selected_model
    ]