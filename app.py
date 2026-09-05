# ============================================================
# STAS GPT - FastAPI Backend
# ============================================================

from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# IMPORTS
# ============================================================

import json
import uuid
import sqlite3
from pathlib import Path

import uvicorn

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
)

from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
)

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

from agent import get_agent

# IMPORTANT:
# Your rag.py contains add_document_to_rag()
from rag import add_document_to_rag


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = BASE_DIR / "generated"

UPLOADS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="STAS GPT",
    description="Agentic AI Chatbot",
    version="1.0.0",
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# ============================================================
# GENERATED DOCUMENTS
# ============================================================
# Generated files are available at:
#
# /generated/example.pdf
# /generated/example.docx
# ============================================================

app.mount(
    "/generated",
    StaticFiles(
        directory=str(GENERATED_DIR)
    ),
    name="generated",
)


# ============================================================
# DATABASE
# ============================================================

DB_PATH = BASE_DIR / "chatbot.db"


def get_db_connection():

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():

    conn = get_db_connection()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Conversations table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Messages table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()

    conn.close()


init_db()


# ============================================================
# CURRENT THREAD
# ============================================================

CURRENT_THREAD_ID = "default"


def set_current_thread_id(
    thread_id: str,
):

    global CURRENT_THREAD_ID

    CURRENT_THREAD_ID = thread_id


def get_current_thread_id():

    return CURRENT_THREAD_ID


# ============================================================
# CREATE / UPDATE CONVERSATION
# ============================================================

def create_or_update_conversation(
    thread_id: str,
    first_message: str = "",
):

    conn = get_db_connection()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Check whether conversation already exists
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT thread_id
        FROM conversations
        WHERE thread_id = ?
        """,
        (
            thread_id,
        ),
    )

    existing = cursor.fetchone()

    # --------------------------------------------------------
    # Existing conversation
    # --------------------------------------------------------

    if existing:

        cursor.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
            """,
            (
                thread_id,
            ),
        )

    # --------------------------------------------------------
    # New conversation
    # --------------------------------------------------------

    else:

        title = first_message.strip()

        if not title:

            title = "New Chat"

        # Keep sidebar title short

        if len(title) > 50:

            title = (
                title[:50]
                + "..."
            )

        cursor.execute(
            """
            INSERT INTO conversations
            (
                thread_id,
                title
            )
            VALUES (?, ?)
            """,
            (
                thread_id,
                title,
            ),
        )

    conn.commit()

    conn.close()


# ============================================================
# SAVE CHAT MESSAGE
# ============================================================

def save_chat_message(
    thread_id: str,
    role: str,
    content: str,
):

    conn = get_db_connection()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Save message
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO chat_messages
        (
            thread_id,
            role,
            content
        )
        VALUES (?, ?, ?)
        """,
        (
            thread_id,
            role,
            content,
        ),
    )

    # --------------------------------------------------------
    # Update conversation timestamp
    # --------------------------------------------------------

    cursor.execute(
        """
        UPDATE conversations
        SET updated_at = CURRENT_TIMESTAMP
        WHERE thread_id = ?
        """,
        (
            thread_id,
        ),
    )

    conn.commit()

    conn.close()


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home(
    request: Request,
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        },
    )


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

@app.get("/conversations")
async def get_conversations():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            thread_id,
            title,
            created_at,
            updated_at
        FROM conversations
        ORDER BY updated_at DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    conversations = []

    for row in rows:

        conversations.append(
            {
                "thread_id": row["thread_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    # ========================================================
    # IMPORTANT FIX
    #
    # Your index.html expects:
    #
    # data.conversations
    #
    # Therefore we return an object containing
    # the conversations list.
    # ========================================================

    return JSONResponse(
        content={
            "conversations": conversations
        }
    )


# ============================================================
# GET CHAT HISTORY
# ============================================================

@app.get("/history/{thread_id}")
async def get_history(
    thread_id: str,
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            role,
            content,
            created_at
        FROM chat_messages
        WHERE thread_id = ?
        ORDER BY id ASC
        """,
        (
            thread_id,
        ),
    )

    rows = cursor.fetchall()

    conn.close()

    history = []

    for row in rows:

        history.append(
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
        )

    # ========================================================
    # IMPORTANT FIX
    #
    # Your index.html expects:
    #
    # data.messages
    #
    # Therefore we return an object containing
    # the messages list.
    # ========================================================

    return JSONResponse(
        content={
            "messages": history
        }
    )


# ============================================================
# DELETE CHAT HISTORY
# ============================================================

@app.delete("/history/{thread_id}")
async def delete_history(
    thread_id: str,
):

    conn = get_db_connection()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Delete messages
    # --------------------------------------------------------

    cursor.execute(
        """
        DELETE FROM chat_messages
        WHERE thread_id = ?
        """,
        (
            thread_id,
        ),
    )

    # --------------------------------------------------------
    # Delete conversation
    # --------------------------------------------------------

    cursor.execute(
        """
        DELETE FROM conversations
        WHERE thread_id = ?
        """,
        (
            thread_id,
        ),
    )

    conn.commit()

    conn.close()

    return JSONResponse(
        content={
            "success": True,
            "message": "Conversation deleted successfully.",
        }
    )


# ============================================================
# FILE UPLOAD
# ============================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    thread_id: str = Form("default"),
):

    try:

        # ----------------------------------------------------
        # Validate filename
        # ----------------------------------------------------

        if not file.filename:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "No file selected.",
                },
            )

        # ----------------------------------------------------
        # Original filename
        # ----------------------------------------------------

        original_name = Path(
            file.filename
        ).name

        extension = Path(
            original_name
        ).suffix.lower()

        # ----------------------------------------------------
        # Supported file types
        # ----------------------------------------------------

        allowed_extensions = {
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".py",
            ".csv",
        }

        if extension not in allowed_extensions:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": (
                        "Unsupported file type. "
                        "Upload PDF, DOCX, TXT, MD, PY, or CSV."
                    ),
                },
            )

        # ----------------------------------------------------
        # Unique filename
        # ----------------------------------------------------

        file_id = str(
            uuid.uuid4()
        )

        saved_filename = (
            f"{file_id}{extension}"
        )

        file_path = (
            UPLOADS_DIR
            / saved_filename
        )

        # ----------------------------------------------------
        # Save file
        # ----------------------------------------------------

        contents = await file.read()

        with open(
            file_path,
            "wb",
        ) as f:

            f.write(contents)

        # ----------------------------------------------------
        # Add file to RAG
        # ----------------------------------------------------

        rag_result = add_document_to_rag(
            str(file_path),
            thread_id,
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "success": True,
                "filename": original_name,
                "saved_filename": saved_filename,
                "thread_id": thread_id,
                "message": (
                    "File uploaded and added "
                    "to RAG successfully."
                ),
                "rag": rag_result,
            }
        )

    except Exception as e:

        print(
            "Upload error:",
            e,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )


# ============================================================
# EXTRACT TEXT FROM AI STREAM
# ============================================================

def extract_text_from_chunk(
    chunk,
) -> str:

    try:

        content = getattr(
            chunk,
            "content",
            "",
        )

        if not content:

            return ""

        # ----------------------------------------------------
        # String
        # ----------------------------------------------------

        if isinstance(
            content,
            str,
        ):

            return content

        # ----------------------------------------------------
        # List
        # ----------------------------------------------------

        if isinstance(
            content,
            list,
        ):

            text_parts = []

            for item in content:

                if isinstance(
                    item,
                    str,
                ):

                    text_parts.append(
                        item
                    )

                elif isinstance(
                    item,
                    dict,
                ):

                    text = item.get(
                        "text"
                    )

                    if text:

                        text_parts.append(
                            str(text)
                        )

            return "".join(
                text_parts
            )

        return str(
            content
        )

    except Exception as e:

        print(
            "Text extraction error:",
            e,
        )

        return ""


# ============================================================
# STREAM FILTER
# ============================================================

def should_stream_chunk(
    chunk,
    metadata,
) -> bool:

    metadata = metadata or {}

    node_name = str(
        metadata.get(
            "langgraph_node",
            "",
        )
    ).lower()

    # --------------------------------------------------------
    # Don't stream tool nodes
    # --------------------------------------------------------

    if "tool" in node_name:

        return False

    # --------------------------------------------------------
    # Don't stream ToolMessage
    # --------------------------------------------------------

    if isinstance(
        chunk,
        ToolMessage,
    ):

        return False

    # --------------------------------------------------------
    # Only AI messages
    # --------------------------------------------------------

    if not isinstance(
        chunk,
        (
            AIMessage,
            AIMessageChunk,
        ),
    ):

        return False

    # --------------------------------------------------------
    # Ignore tool calls
    # --------------------------------------------------------

    if getattr(
        chunk,
        "tool_calls",
        None,
    ):

        return False

    if getattr(
        chunk,
        "invalid_tool_calls",
        None,
    ):

        return False

    additional_kwargs = getattr(
        chunk,
        "additional_kwargs",
        {},
    ) or {}

    if additional_kwargs.get(
        "tool_calls"
    ):

        return False

    return True


# ============================================================
# EXTRACT GENERATED DOCUMENT
# ============================================================

def extract_generated_document(
    tool_message,
):
    """
    Extract document information from
    create_document ToolMessage.

    Expected tool response:

    {
        "success": true,
        "type": "pdf",
        "filename": "example.pdf",
        "download_url": "/generated/example.pdf"
    }
    """

    try:

        content = getattr(
            tool_message,
            "content",
            "",
        )

        if not content:

            return None

        # ----------------------------------------------------
        # JSON string
        # ----------------------------------------------------

        if isinstance(
            content,
            str,
        ):

            try:

                data = json.loads(
                    content
                )

            except Exception:

                return None

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        elif isinstance(
            content,
            dict,
        ):

            data = content

        else:

            return None

        # ----------------------------------------------------
        # Check success
        # ----------------------------------------------------

        if not data.get(
            "success",
            False,
        ):

            return None

        # ----------------------------------------------------
        # Get file information
        # ----------------------------------------------------

        download_url = data.get(
            "download_url"
        )

        filename = data.get(
            "filename"
        )

        file_type = data.get(
            "type",
            "",
        )

        if not download_url:

            return None

        if not filename:

            return None

        # ----------------------------------------------------
        # Frontend-compatible structure
        # ----------------------------------------------------

        return {
            "url": download_url,
            "filename": filename,
            "type": file_type,
        }

    except Exception as e:

        print(
            "Document extraction error:",
            e,
        )

        return None


# ============================================================
# SSE FORMAT
# ============================================================

def sse_data(
    data: dict,
) -> str:

    return (
        f"data: {json.dumps(data)}\n\n"
    )


# ============================================================
# CHAT STREAM
# ============================================================

@app.post("/chat/stream")
async def chat_stream(
    request: Request,
):

    # --------------------------------------------------------
    # Read request
    # --------------------------------------------------------

    try:

        body = await request.json()

    except Exception:

        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid JSON request."
            },
        )

    # --------------------------------------------------------
    # Get values
    # --------------------------------------------------------

    user_message = body.get(
        "message",
        "",
    )

    thread_id = body.get(
        "thread_id"
    )

    selected_model = body.get(
        "model"
    )

    # --------------------------------------------------------
    # Validate message
    # --------------------------------------------------------

    if not user_message.strip():

        return JSONResponse(
            status_code=400,
            content={
                "error": "Message cannot be empty."
            },
        )

    # --------------------------------------------------------
    # Create thread
    # --------------------------------------------------------

    if not thread_id:

        thread_id = str(
            uuid.uuid4()
        )

    # ========================================================
    # CREATE / UPDATE CONVERSATION
    # ========================================================

    create_or_update_conversation(
        thread_id,
        user_message,
    )

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    save_chat_message(
        thread_id,
        "user",
        user_message,
    )

    # ========================================================
    # SET CURRENT THREAD
    # ========================================================

    set_current_thread_id(
        thread_id
    )

    # ========================================================
    # GET AGENT
    # ========================================================

    try:

        agent = get_agent(
            selected_model
        )

    except Exception as e:

        print(
            "Agent creation error:",
            e,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            },
        )

    # ========================================================
    # LANGGRAPH CONFIG
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # ========================================================
    # EVENT GENERATOR
    # ========================================================

    def event_generator():

        final_answer = ""

        try:

            inputs = {
                "messages": [
                    HumanMessage(
                        content=user_message
                    )
                ]
            }

            # =================================================
            # STREAM LANGGRAPH
            # =================================================

            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages",
            ):

                # =============================================
                # TOOL MESSAGE
                # =============================================

                if isinstance(
                    chunk,
                    ToolMessage,
                ):

                    document_info = (
                        extract_generated_document(
                            chunk
                        )
                    )

                    # -----------------------------------------
                    # Generated PDF / DOCX
                    # -----------------------------------------

                    if document_info:

                        yield sse_data(
                            {
                                "document": document_info
                            }
                        )

                    # Don't send raw tool output
                    continue

                # =============================================
                # NORMAL AI MESSAGE
                # =============================================

                if not should_stream_chunk(
                    chunk,
                    metadata,
                ):

                    continue

                token = extract_text_from_chunk(
                    chunk
                )

                if token:

                    final_answer += token

                    yield sse_data(
                        {
                            "token": token
                        }
                    )

            # =================================================
            # SAVE ASSISTANT RESPONSE
            # =================================================

            if final_answer.strip():

                save_chat_message(
                    thread_id,
                    "assistant",
                    final_answer,
                )

            # =================================================
            # UPDATE CONVERSATION TIMESTAMP
            # =================================================

            create_or_update_conversation(
                thread_id
            )

            # =================================================
            # FINISHED
            # =================================================

            yield sse_data(
                {
                    "done": True,
                    "thread_id": thread_id,
                }
            )

        except Exception as e:

            print(
                "Chat stream error:",
                e,
            )

            yield sse_data(
                {
                    "error": str(e)
                }
            )

            yield sse_data(
                {
                    "done": True,
                    "thread_id": thread_id,
                }
            )

    # ========================================================
    # RETURN SSE
    # ========================================================

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "STAS GPT",
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )