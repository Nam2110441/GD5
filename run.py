from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingestion.ingestion import Ingestion
from chatbot.services.files_chat_agent import FilesChatAgent
from app.config import settings
from models.character_prompts import CharacterPrompts

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    character: str | None = None  # Cho phép rỗng hoặc không truyền

@app.post("/chat")
def chat_with_character(request: ChatRequest):
    """
    API nhận câu hỏi và nhân vật, nếu không có nhân vật, chatbot sẽ trả lời thông thường.
    """
    if request.character:
        character_prompt = CharacterPrompts.get_prompt(request.character)
        if character_prompt is None:
            raise HTTPException(status_code=400, detail="Nhân vật không hợp lệ")
        prompt = f"{character_prompt}\n\n{request.question}"
    else:
        prompt = request.question  # Không thêm phong cách nhân vật

    chat_agent = FilesChatAgent("demo/data_vector").get_workflow().compile()

    response = chat_agent.invoke(input={"question": prompt})

    return {
        "character": request.character if request.character else "normal",
        "question": request.question,
        "answer": response["generation"],
    }
