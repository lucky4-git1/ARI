"""Local HTTP API using FastAPI for LAYA-LLM inference."""

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import torch

from layallm.model.config import ModelConfig
from layallm.model.transformer import LayaTransformer
from layallm.tokenizer.tokenizer import LayaTokenizer
from layallm.training.checkpoint import load_checkpoint
from layallm.inference.generate import CPUInferenceEngine

app = FastAPI(title="LAYA-LLM Inference API", version="0.1.0")

# Global engine container
engine: Optional[CPUInferenceEngine] = None


class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9


class GenerationResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int


class ModelInfoResponse(BaseModel):
    status: str
    vocab_size: int
    parameters: int
    hidden_dim: int
    num_layers: int
    device: str


def initialize_api(checkpoint_path: str, tokenizer_path: str):
    """Loads checkpoint and tokenizer into memory for serving."""
    global engine
    tokenizer = LayaTokenizer.load(tokenizer_path)
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    model = loaded["model"]
    engine = CPUInferenceEngine(model, tokenizer, num_threads=4)


@app.get("/health")
def health():
    if engine is None:
        return {"status": "uninitialized"}
    return {"status": "ready"}


@app.get("/info", response_model=ModelInfoResponse)
def get_info():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    counts = engine.model.count_parameters()
    cfg = engine.model.config
    return ModelInfoResponse(
        status="active",
        vocab_size=cfg.vocab_size,
        parameters=counts["total_parameters"],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        device="cpu",
    )


@app.post("/generate", response_model=GenerationResponse)
def generate_text(req: GenerationRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    prompt_ids = engine.tokenizer.encode(req.prompt)
    out_text = engine.generate(
        req.prompt,
        max_new_tokens=min(req.max_new_tokens, 512),
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    total_ids = engine.tokenizer.encode(out_text)
    
    return GenerationResponse(
        text=out_text,
        prompt_tokens=len(prompt_ids),
        completion_tokens=max(0, len(total_ids) - len(prompt_ids)),
    )
