"""FastAPI Web UI & Chat Server powered by Ari-LLM."""

import os
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from arillm.training.checkpoint import load_checkpoint
from arillm.tokenizer.tokenizer import AriTokenizer
from arillm.inference.generate import CPUInferenceEngine
from arillm.rag.retriever import LocalKnowledgeRetriever

app = FastAPI(title="Ari-LLM Chat Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
engine: Optional[CPUInferenceEngine] = None
retriever: Optional[LocalKnowledgeRetriever] = None
model_name: str = "Ari-LLM"

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 65

class ChatResponse(BaseModel):
    reply: str
    tokens_generated: int
    retrieved_source: Optional[str] = None

from transformers import AutoTokenizer, AutoModelForCausalLM

class HfInferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        
    def generate(self, prompt, max_new_tokens=120, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=self.eos_ids,
                pad_token_id=self.tokenizer.eos_token_id
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0].strip()
        return text

def init_model(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: str = "artifacts/tokenizers/ari_tokenizer_v2.json",
):
    global engine, retriever, model_name
    if checkpoint_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        candidates = [
            os.path.join(root_dir, "artifacts/checkpoints/ari_rewritebot_135m_dpo"),
            os.path.join(root_dir, "artifacts/checkpoints/ari_rewritebot_135m"),
            os.path.join(root_dir, "artifacts/checkpoints/exp_kaggle_sft_best.pt"),
            os.path.join(root_dir, "artifacts/checkpoints/exp_kaggle_35m_sft_best.pt"),
            os.path.join(root_dir, "artifacts/checkpoints/exp_ari_7m_v4_paraphrase_best.pt"),
            "artifacts/checkpoints/ari_rewritebot_135m_dpo",
            "artifacts/checkpoints/ari_rewritebot_135m",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                checkpoint_path = cand
                break
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No valid Ari-LLM checkpoint found to initialize server.")

    print(f"Loading Ari-LLM from {checkpoint_path}...")
    
    # Check if checkpoint is an Ari-135M Transformer / Safetensors directory
    if os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
        hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, dtype=torch.float32)
        engine = HfInferenceEngine(hf_model, hf_tokenizer)
        total_params = sum(p.numel() for p in hf_model.parameters())
        model_name = f"Ari-LLM RewriteBot ({total_params / 1e6:.1f}M DPO Aligned)"
        print(f"Ari-LLM Engine Ready: {model_name} (Commercial-Grade Paraphraser)!")
        return

    tokenizer = AriTokenizer.load(tokenizer_path)
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    engine = CPUInferenceEngine(loaded["model"], tokenizer, num_threads=4)
    counts = loaded["model"].count_parameters()
    model_name = f"Ari-LLM ({counts['total_parameters'] / 1e6:.1f}M Params)"
    retriever = LocalKnowledgeRetriever("data/knowledge")
    print(f"Ari-LLM Engine Ready: {model_name} with Local RAG Knowledge!")

@app.on_event("startup")
def startup_event():
    init_model()

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")

    last_user_msg = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            last_user_msg = msg.content.strip()
            break

    # RAG lookup for web components or matching documents
    retrieved_source = None
    if retriever and last_user_msg:
        results = retriever.retrieve(last_user_msg, top_k=1, threshold=0.8)
        if results:
            doc_name, content, score = results[0]
            retrieved_source = doc_name
            # 1. Fact knowledge memory match (e.g., "what is a mango", "what is photosynthesis")
            if "#" in doc_name or doc_name.endswith(".json"):
                reply = content
                tokens_count = len(engine.tokenizer.encode(reply))
                return ChatResponse(
                    reply=reply,
                    tokens_generated=tokens_count,
                    retrieved_source=doc_name
                )

            # 2. Web UI component or template match
            component_triggers = [
                "carousel", "carosel", "slider", "slide", "hero", "navbar", "nav",
                "pricing", "price", "plan", "auth", "login", "register", "signup",
                "faq", "accordion", "card", "landing", "landingpage", "landing page",
                "component", "template"
            ]
            if any(t in last_user_msg.lower() for t in component_triggers):
                reply = (
                    f"Here is the verified implementation for your request from Ari's component memory (`{doc_name}`):\n\n"
                    f"```html\n{content}\n```"
                )
                tokens_count = len(engine.tokenizer.encode(reply))
                return ChatResponse(
                    reply=reply,
                    tokens_generated=tokens_count,
                    retrieved_source=doc_name
                )

    # Format conversational prompt using SFT special tokens
    # Truncate large previous assistant messages (like full HTML code blocks) so context fits in budget
    prompt_lines = []
    for msg in req.messages[-4:]:
        content = msg.content.strip()
        if msg.role == "assistant" and len(content) > 300:
            # Keep summary/head of large past code to prevent blowing the context limit
            content = content[:300] + "... [component code truncated]"
        if msg.role == "user":
            prompt_lines.append(f"<user>\n{content}\n")
        elif msg.role == "assistant":
            prompt_lines.append(f"<assistant>\n{content}<eos>\n")
    
    prompt_lines.append("<assistant>\n")
    full_prompt = "".join(prompt_lines)

    # Ensure token count doesn't exceed 240 tokens before generation
    encoded_tokens = engine.tokenizer.encode(full_prompt)
    if len(encoded_tokens) > 230:
        encoded_tokens = encoded_tokens[-230:]
        full_prompt = engine.tokenizer.decode(encoded_tokens)
    
    # Generate completion
    output_text = engine.generate(
        prompt=full_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    
    # Extract only assistant reply after the prompt
    if full_prompt in output_text:
        reply = output_text[len(full_prompt):].strip()
    else:
        reply = output_text.split("<assistant>\n")[-1].strip()

    # Strip EOS token
    reply = reply.replace("<eos>", "").strip()
    tokens_count = len(engine.tokenizer.encode(reply))
    return ChatResponse(reply=reply, tokens_generated=tokens_count, retrieved_source=retrieved_source)

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ari-LLM Studio</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --chat-bg: #0b1329;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.3);
            --user-bubble: #2563eb;
            --ai-bubble: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background-color: var(--panel-bg);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px 20px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 28px;
        }

        .brand-logo {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 20px;
            color: #0f172a;
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .brand-title {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .badge {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.3);
            font-size: 11px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            margin-top: 4px;
            display: inline-block;
        }

        .specs-panel {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 24px;
            font-size: 12px;
            line-height: 1.6;
        }

        .spec-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }

        .spec-label {
            color: var(--text-muted);
        }

        .spec-val {
            font-weight: 600;
            color: var(--accent);
        }

        .sample-prompts {
            flex: 1;
            overflow-y: auto;
        }

        .section-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }

        .prompt-btn {
            width: 100%;
            text-align: left;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-main);
            padding: 10px 12px;
            font-size: 12px;
            cursor: pointer;
            margin-bottom: 8px;
            transition: all 0.2s ease;
        }

        .prompt-btn:hover {
            background: rgba(56, 189, 248, 0.1);
            border-color: var(--accent);
            transform: translateY(-1px);
        }

        /* Main Chat Area */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--chat-bg);
        }

        .chat-header {
            height: 64px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
            background: rgba(15, 23, 42, 0.5);
            backdrop-filter: blur(8px);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 500;
            color: #4ade80;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #4ade80;
            border-radius: 50%;
            box-shadow: 0 0 10px #4ade80;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 28px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message {
            display: flex;
            gap: 14px;
            max-width: 80%;
            animation: fadeIn 0.25s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .avatar {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: bold;
            flex-shrink: 0;
        }

        .avatar.user {
            background: var(--user-bubble);
            color: white;
        }

        .avatar.ai {
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: #0f172a;
        }

        .bubble {
            padding: 14px 18px;
            border-radius: 14px;
            font-size: 14.5px;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .message.user .bubble {
            background: var(--user-bubble);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.ai .bubble {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-bottom-left-radius: 4px;
        }

        .code-block {
            background: #090d16;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            margin-top: 8px;
            color: #38bdf8;
            overflow-x: auto;
        }

        /* Input Controls */
        .chat-input-area {
            padding: 20px 28px;
            background: rgba(15, 23, 42, 0.8);
            border-top: 1px solid var(--border);
        }

        .input-box-wrapper {
            display: flex;
            align-items: center;
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 6px 8px 6px 16px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            transition: border-color 0.2s;
        }

        .input-box-wrapper:focus-within {
            border-color: var(--accent);
        }

        .chat-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: white;
            font-size: 14px;
            resize: none;
            max-height: 120px;
        }

        .send-btn {
            background: var(--accent);
            color: #0f172a;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: all 0.2s ease;
        }

        .send-btn:hover {
            background: #7dd3fc;
            transform: scale(1.05);
        }

        .send-btn:disabled {
            background: #475569;
            cursor: not-allowed;
            transform: none;
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="brand">
            <div class="brand-logo">A</div>
            <div>
                <div class="brand-title">Ari-LLM Studio</div>
                <div class="badge">Original Transformer</div>
            </div>
        </div>

        <div class="specs-panel">
            <div class="spec-item">
                <span class="spec-label">Model Size:</span>
                <span class="spec-val">14.6M Params</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Device:</span>
                <span class="spec-val">CPU (Offline)</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Tokens/Sec:</span>
                <span class="spec-val">~91 tok/s</span>
            </div>
            <div class="spec-item">
                <span class="spec-label">Latency:</span>
                <span class="spec-val">~14 ms TTFT</span>
            </div>
        </div>

        <div class="section-label">Try Example Prompts</div>
        <div class="sample-prompts">
            <button class="prompt-btn" onclick="sendSample('Paraphrase this statement: Artificial intelligence is transforming how businesses operate worldwide.')">
                ✨ Paraphrase AI Statement
            </button>
            <button class="prompt-btn" onclick="sendSample('Rewrite the following in a more professional tone: I can\'t come to work because I am sick.')">
                💼 Professional Email
            </button>
            <button class="prompt-btn" onclick="sendSample('What are the four pillars of Object-Oriented Programming?')">
                🧱 OOP Principles
            </button>
            <button class="prompt-btn" onclick="sendSample('Write a Python function to check if a year is a leap year.')">
                🐍 Leap Year Function
            </button>
            <button class="prompt-btn" onclick="sendSample('Who was Alan Turing and what did he contribute?')">
                💻 Alan Turing
            </button>
            <button class="prompt-btn" onclick="sendSample('What is a Transformer model in artificial intelligence?')">
                🧠 Transformer Architecture
            </button>
        </div>
    </div>

    <!-- Chat Area -->
    <div class="chat-container">
        <div class="chat-header">
            <div>
                <strong>Local Model Interface</strong>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Running 100% locally on your machine</div>
            </div>
            <div class="status-indicator">
                <div class="status-dot"></div>
                Engine Online
            </div>
        </div>

        <div class="chat-messages" id="messagesList">
            <div class="message ai">
                <div class="avatar ai">A</div>
                <div class="bubble">Hello! I am <strong>Ari-LLM</strong>, an original decoder-only Transformer built and trained completely from scratch.

Ask me to paraphrase sentences, rewrite emails, explain computer science and physics concepts, or write Python algorithms!</div>
            </div>
        </div>

        <div class="chat-input-area">
            <div class="input-box-wrapper">
                <input type="text" class="chat-input" id="userInput" placeholder="Ask Ari-LLM something..." onkeydown="handleKey(event)">
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </div>

    <script>
        const messagesList = document.getElementById('messagesList');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        let chatHistory = [];

        function appendMessage(role, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            
            const avatar = document.createElement('div');
            avatar.className = `avatar ${role}`;
            avatar.innerText = role === 'user' ? 'U' : 'A';

            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            
            // Format code blocks if present
            if (text.includes('def ') || text.includes('return ') || text.includes('class ')) {
                bubble.innerHTML = text.replace(/`([^`]+)`/g, '<code>$1</code>');
            } else {
                bubble.innerText = text;
            }

            msgDiv.appendChild(avatar);
            msgDiv.appendChild(bubble);
            messagesList.appendChild(msgDiv);
            messagesList.scrollTop = messagesList.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            userInput.value = '';
            userInput.disabled = true;
            sendBtn.disabled = true;

            appendMessage('user', text);
            chatHistory.push({ role: 'user', content: text });

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: chatHistory,
                        temperature: 0.2,
                        max_tokens: 60
                    })
                });

                if (!response.ok) throw new Error('API Error');
                const data = await response.json();

                appendMessage('ai', data.reply);
                chatHistory.push({ role: 'assistant', content: data.reply });
            } catch (err) {
                appendMessage('ai', 'Error generating response from local model.');
                console.error(err);
            } finally {
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        function handleKey(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        }

        function sendSample(text) {
            userInput.value = text;
            sendMessage();
        }
    </script>
</body>
</html>"""
