from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.models import QuestionRequest, QuestionResponse
from app.services.qa_service import qa_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    print("Loading messages...")
    await qa_service.initialize()
    print("Ready!")
    yield

app = FastAPI(title="Member Q&A System", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Member Q&A API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "messages_loaded": len(qa_service.messages)}

@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """Ask a question about member data"""
    try:
        answer = await qa_service.answer(request.question)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

