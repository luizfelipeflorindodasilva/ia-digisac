import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, FileResponse
from groq import Groq
from pydantic import BaseModel

import db

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY não definida. Crie o arquivo .env com sua chave.")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Carrega base de conhecimento (chunks da API Digisac)
# ---------------------------------------------------------------------------
CHUNKS_PATH = Path(__file__).parent / "digisac_chunks.json"
if not CHUNKS_PATH.exists():
    raise RuntimeError(f"Arquivo de chunks não encontrado: {CHUNKS_PATH}")

with open(CHUNKS_PATH, encoding="utf-8") as f:
    API_CHUNKS: list[dict] = json.load(f)

# Também carrega a documentação completa (usada como fallback / visão geral)
DOCS_PATH = Path(__file__).parent / "digisac_docs.txt"
DIGISAC_DOCS_FULL = DOCS_PATH.read_text(encoding="utf-8") if DOCS_PATH.exists() else ""

# ---------------------------------------------------------------------------
# Mapeamento de aliases e palavras-chave por grupo temático
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    "autorizacao": ["token", "autorizac", "login", "oauth", "senha", "acesso", "autenticar"],
    "contatos": ["contato", "contact", "número", "numero", "cliente", "bloquear", "sincroniz",
                 "cadastrado", "não cadastrado", "nao cadastrado"],
    "mensagens": ["mensagem", "message", "enviar", "chat", "texto", "imagem", "audio", "áudio",
                  "pdf", "arquivo", "reagir", "reação", "comentar", "comment",
                  "cadastrado", "não cadastrado", "nao cadastrado", "number", "serviceid"],
    "chamados": ["chamado", "ticket", "abrir", "fechar", "transferir", "transferência",
                 "protocolo", "atendimento"],
    "departamentos": ["departamento", "department"],
    "usuarios": ["usuário", "usuario", "user", "atendente"],
    "agendamentos": ["agendamento", "schedule", "agendar"],
    "webhooks": ["webhook", "notificação", "evento", "event"],
    "tags": ["tag", "etiqueta"],
    "templates": ["template", "hsm", "whatsapp business", "waba"],
    "campanhas": ["campanha", "disparo", "massa"],
    "conexoes": ["conexão", "conexao", "service", "qr code", "reiniciar", "desligar"],
    "bots": ["bot", "robô", "robo", "chatbot", "flag"],
    "estatisticas": ["estatística", "estatistica", "dashboard", "relatório", "relatorio"],
    "grupos": ["grupo", "group", "participante", "administrador"],
    "campos": ["campo personalizado", "custom field"],
    "cargos": ["cargo", "role", "permissão", "permissao"],
    "rapidas": ["resposta rápida", "resposta rapida", "quick reply"],
    "interativas": ["mensagem interativa", "botão", "lista interativa", "interactive"],
    "pessoas": ["pessoa", "people", "organização", "organizacao"],
    "tokens": ["token pessoal", "me/tokens"],
    "agora": ["agora", "tempo real", "now", "resumo"],
}


def _strip_accents(s: str) -> str:
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def search_chunks(query: str, max_chunks: int = 30) -> list[dict]:
    """
    Busca os chunks mais relevantes para a query do usuário.
    Pontuação ponderada com normalização de acentos para evitar falsos negativos.
    """
    query_lower = query.lower()
    query_norm = _strip_accents(query_lower)
    query_words = set(re.findall(r'\w+', query_norm))

    scored: list[tuple[int, dict]] = []

    for chunk in API_CHUNKS:
        score = 0
        name_norm = _strip_accents(chunk["name"].lower())
        kw_norm = _strip_accents(chunk["keywords"])
        body_norm = _strip_accents(chunk.get("body", "").lower())

        # Correspondência exata de substring da query no nome do endpoint
        if query_norm in name_norm:
            score += 5

        for w in query_words:
            if len(w) < 3:
                continue
            if w in name_norm.split():
                score += 4         # palavra exata no nome
            elif w in name_norm:
                score += 2         # substring no nome
            if w in kw_norm:
                score += 1         # palavra nas keywords
            if w in body_norm:
                score += 2         # palavra no body do endpoint

        # Boost por grupo temático identificado na query
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw_topic in keywords:
                kt_norm = _strip_accents(kw_topic)
                if kt_norm in query_norm and kt_norm in kw_norm:
                    score += 3

        # Boost pelo método HTTP quando mencionado explicitamente
        for method in ["get", "post", "put", "delete", "patch"]:
            if method in query_norm and chunk["method"].lower() == method:
                score += 2

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in scored[:max_chunks]]

    # Garante que todos os chunks do grupo Mensagens sejam incluídos quando
    # a pergunta for sobre envio de mensagens (evita omitir variantes do mesmo endpoint)
    if any(w in query_norm for w in ["mensagem", "enviar", "message"]):
        extras = [c for c in API_CHUNKS
                  if _strip_accents(c["group"].lower()) == "mensagens" and c not in result]
        result = result + extras

    return result


def build_context(chunks: list[dict]) -> str:
    """Formata os chunks encontrados como contexto para o LLM."""
    if not chunks:
        return "Nenhum endpoint específico encontrado para essa consulta."

    lines = []
    current_group = None
    for c in chunks:
        if c["group"] != current_group:
            current_group = c["group"]
            lines.append(f"\n### {current_group}")
        lines.append(f"  [{c['method']}] {c['name']}")
        lines.append(f"  Endpoint: {c['url']}")
        if c["body"]:
            lines.append(f"  Body:")
            for bl in c["body"].split("\n")[:15]:
                lines.append(f"    {bl}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt base do sistema
# ---------------------------------------------------------------------------
SYSTEM_CURTO = """Você é um assistente de suporte técnico da API Digisac.

Responda APENAS sobre a API Digisac. Para qualquer outro assunto, recuse educadamente.

REGRA CRÍTICA: Se a funcionalidade solicitada não estiver presente na DOCUMENTAÇÃO RELEVANTE fornecida abaixo, responda EXATAMENTE com esta frase e nada mais:
Não encontrei documentação para isso na API Digisac. Essa funcionalidade pode não existir na plataforma ou a requisição ainda não foi desenvolvida.

Se a funcionalidade ESTIVER documentada, sua resposta deve conter SOMENTE:
1. O(s) comando(s) curl prontos para copiar e colar no Postman
2. A pergunta final fixa

Use este formato exato — sem títulos, sem explicações, sem introduções:

Aqui está o comando pronto para [descrição de 5 palavras no máximo]:

```
curl -X MÉTODO \\
  {{URL}}/api/v1/caminho \\
  -H 'Authorization: Bearer {{token}}' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "campo1": "{{valor1}}",
    "campo2": "{{valor2}}"
}'
```

O body DEVE sempre estar formatado com cada campo em uma linha separada, com indentação de 4 espaços, exatamente como no exemplo acima. Nunca coloque o body em uma única linha.

Se houver variantes reais com bodies diferentes, mostre TODAS — não omita nenhuma. Apresente assim:
Opção 1 — nome curto:
```
curl -X MÉTODO \\
  {{URL}}/api/v1/caminho \\
  -H 'Authorization: Bearer {{token}}' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "campo": "{{valor}}"
}'
```
Opção 2 — nome curto:
```
curl ...
```

Termine SEMPRE com esta linha exata:
Ficou com alguma dúvida ou quer uma explicação mais detalhada de como configurar isso no Postman?

INFORMAÇÕES:
- URL base: {{URL}} = https://SEU-SUBDOMINIO.digisac.co
- Header obrigatório: Authorization: Bearer {{token}}
- Nunca invente variantes que não existam na documentação.
"""

SYSTEM_DETALHADO = """Você é um assistente de suporte técnico da API Digisac.

Responda APENAS sobre a API Digisac. Para qualquer outro assunto, recuse educadamente.

REGRA CRÍTICA: Se a funcionalidade solicitada não estiver presente na DOCUMENTAÇÃO RELEVANTE fornecida abaixo, responda EXATAMENTE com esta frase e nada mais:
Não encontrei documentação para isso na API Digisac. Essa funcionalidade pode não existir na plataforma ou a requisição ainda não foi desenvolvida.

Se a funcionalidade ESTIVER documentada, o usuário pediu uma explicação mais detalhada. Envie a resposta completa em duas partes:

PARTE 1 — Passo a passo no Postman (linguagem simples, como se a pessoa nunca tivesse usado uma API):
Passo 1: ...
Passo 2: ...
Se houver variantes com bodies diferentes, use "--- Opção 1: nome ---" e "--- Opção 2: nome ---" com passos separados para cada uma.

PARTE 2 — Escreva exatamente este texto e depois o JSON:
Para realizar de forma mais simples, copie e cole esse código no seu Postman, preencha os campos das variáveis e faça a requisição.

```json
{
  "info": { "name": "NOME", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json" },
  "item": [
    {
      "name": "NOME",
      "request": {
        "method": "MÉTODO",
        "header": [
          { "key": "Authorization", "value": "Bearer {{token}}", "type": "text" },
          { "key": "Content-Type", "value": "application/json", "type": "text" }
        ],
        "url": { "raw": "{{URL}}/api/v1/...", "host": ["{{URL}}"], "path": ["api","v1","..."] },
        "body": { "mode": "raw", "raw": "{\n    \"campo1\": \"{{valor1}}\",\n    \"campo2\": \"{{valor2}}\"\n}", "options": { "raw": { "language": "json" } } }
      }
    }
  ]
}
```
Se houver múltiplas variantes, adicione cada uma como item separado dentro de "item".

INFORMAÇÕES:
- URL base: {{URL}} = https://SEU-SUBDOMINIO.digisac.co
- Header obrigatório: Authorization: Bearer {{token}}
- Nunca invente variantes que não existam na documentação.
- O endpoint POST /api/v1/messages tem variantes reais: contato cadastrado (contactId), não cadastrado (number + serviceId), sem abrir chamado (dontOpenTicket), via bot, com mídia.
"""


# ---------------------------------------------------------------------------
# Sessões (histórico de conversa)
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict]] = {}
MAX_HISTORY = 4

# Palavras que indicam que o usuário quer resposta detalhada
DETALHES_TRIGGERS = [
    "sim", "quero", "pode", "detalh", "complet", "passo", "explica",
    "como", "ajud", "mais", "entend", "não entendi", "nao entendi",
    "duvid", "dúvid", "confus", "perdid",
]

def quer_detalhes(msg: str) -> bool:
    m = msg.lower().strip()
    return any(t in m for t in DETALHES_TRIGGERS)


# ---------------------------------------------------------------------------
# Admin tokens (em memória, expiram em 8h)
# ---------------------------------------------------------------------------
admin_tokens: dict[str, datetime] = {}
ADMIN_TOKEN_TTL = timedelta(hours=8)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _generate_admin_token() -> str:
    token = str(uuid.uuid4())
    admin_tokens[token] = datetime.now(timezone.utc)
    return token


def _validate_admin_token(token: str) -> bool:
    if token not in admin_tokens:
        return False
    created_at = admin_tokens[token]
    if datetime.now(timezone.utc) - created_at > ADMIN_TOKEN_TTL:
        del admin_tokens[token]
        return False
    return True


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class AdminLoginRequest(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="IA Suporte Digisac",
    description="Assistente de dúvidas sobre a API Digisac, baseado em Groq + LLaMA.",
    version="1.0.0",
)

@app.on_event("startup")
def startup_event():
    db.init_db()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>IA Suporte Digisac</h1>")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    html_path = Path(__file__).parent / "static" / "admin.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Painel Admin</h1>", status_code=404)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Decide se é resposta curta (curl) ou detalhada (passo a passo + JSON)
    ultima_resposta = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    modo_detalhado = (
        "Ficou com alguma dúvida" in ultima_resposta
        and quer_detalhes(req.message)
    )

    system_prompt_base = SYSTEM_DETALHADO if modo_detalhado else SYSTEM_CURTO
    max_tokens = 1800 if modo_detalhado else 1200
    model_override = GROQ_MODEL if modo_detalhado else "llama-3.1-8b-instant"
    max_chunks = 25 if modo_detalhado else 12

    # Busca chunks relevantes
    relevant_chunks = search_chunks(
        req.message if not modo_detalhado else ultima_resposta + " " + req.message,
        max_chunks=max_chunks
    )
    context = build_context(relevant_chunks)

    system_prompt = (
        system_prompt_base
        + "\n\nDOCUMENTAÇÃO RELEVANTE:\n---\n"
        + context
        + "\n---"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": req.message})

    # Modelos de fallback usados em ordem quando o principal atinge o limite
    FALLBACK_MODELS = [
        model_override,
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
    ]
    # Remove duplicatas mantendo ordem
    seen_models: set = set()
    FALLBACK_MODELS = [m for m in FALLBACK_MODELS if not (m in seen_models or seen_models.add(m))]

    answer = None
    last_error = None

    for model in FALLBACK_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            answer = completion.choices[0].message.content
            break
        except Exception as e:
            err_str = str(e)
            last_error = err_str
            if "429" in err_str or "rate_limit" in err_str:
                continue
            raise HTTPException(status_code=500, detail=f"Erro ao chamar a API Groq: {err_str}")

    # Se a resposta for a mensagem de "não encontrei", trunca qualquer texto extra
    NOT_FOUND_MSG = "Não encontrei documentação para isso na API Digisac. Essa funcionalidade pode não existir na plataforma ou a requisição ainda não foi desenvolvida."
    if answer and answer.strip().startswith("Não encontrei documentação"):
        answer = NOT_FOUND_MSG

    if answer is None:
        wait = ""
        import re as _re
        match = _re.search(r'try again in ([\w\d.]+)', last_error or "")
        if match:
            wait = f" Tente novamente em aproximadamente {match.group(1)}."
        raise HTTPException(
            status_code=429,
            detail=(
                f"O limite diário de requisições foi atingido em todos os modelos disponíveis.{wait} "
                "Isso ocorre no plano gratuito da Groq (100 mil tokens/dia). "
                "O limite é renovado automaticamente todo dia."
            )
        )

    # Salva no histórico (sem o system, só user/assistant)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})

    # Persiste no banco
    db.save_message(session_id, "user", req.message)
    db.save_message(session_id, "assistant", answer)

    return ChatResponse(response=answer, session_id=session_id)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Sessão encerrada."}


@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL, "chunks": len(API_CHUNKS)}


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------
@app.post("/admin/login")
async def admin_login(req: AdminLoginRequest):
    if not ADMIN_EMAIL or not ADMIN_PASSWORD_HASH:
        raise HTTPException(status_code=503, detail="Admin não configurado.")
    if req.email != ADMIN_EMAIL:
        raise HTTPException(status_code=401, detail="Credenciais inválidas.")
    if _hash_password(req.password) != ADMIN_PASSWORD_HASH:
        raise HTTPException(status_code=401, detail="Credenciais inválidas.")
    token = _generate_admin_token()
    return {"token": token}


@app.get("/admin/logs")
async def admin_logs(authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "").strip()
    if not _validate_admin_token(token):
        raise HTTPException(status_code=401, detail="Token inválido ou expirado.")
    conversations = db.get_all_conversations()
    return {"conversations": conversations}


@app.get("/admin/logs/{session_id}")
async def admin_session_messages(session_id: str, authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "").strip()
    if not _validate_admin_token(token):
        raise HTTPException(status_code=401, detail="Token inválido ou expirado.")
    messages = db.get_conversation_messages(session_id)
    return {"session_id": session_id, "messages": messages}


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
