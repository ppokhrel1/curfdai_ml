
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import torch
from app.core.rag_orchestrator import SelfHostedRAGSystem

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = SelfHostedRAGSystem()
    yield
    await app.state.rag.model_manager.unload_all()

app = FastAPI(title="Multi-Model RAG API", lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health():
    rag = app.state.rag
    status = await rag.model_manager.get_status()
    return {
        "gpu": torch.cuda.is_available(),
        "vram_free_gb": f"{status['vram_free_mb'] / 1024:.1f}",
        "loaded_model": status["loaded_model"]
    }

@app.post("/generate-model")
async def generate_model(req: GenerateRequest):
    try:
        result = await app.state.rag.generate_complete_model(req.prompt)
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    path: str = Form(...)
):
    content = await file.read()
    url = await app.state.rag.supabase.upload(content, path, file.content_type)
    return {"url": url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)