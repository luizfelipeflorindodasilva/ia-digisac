import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from groq import Groq
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
SYSTEM_BASE = """Você é um assistente especialista em suporte técnico da API Digisac.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS perguntas relacionadas à API Digisac — endpoints, parâmetros, autenticação, exemplos de requisição/resposta e funcionamento dos recursos.
2. Se a pergunta for sobre qualquer outro assunto (política, programação genérica, assuntos pessoais, etc.), recuse educadamente e oriente o usuário a perguntar sobre a API Digisac.
3. Suas respostas devem ser baseadas EXCLUSIVAMENTE na documentação fornecida abaixo. Não invente endpoints ou parâmetros.
4. Responda sempre em português brasileiro.
5. IMPORTANTE — VARIANTES DO MESMO ENDPOINT: Um mesmo endpoint (ex: POST /api/v1/messages) pode ter múltiplos bodies diferentes para casos de uso distintos. Quando isso ocorrer, apresente TODAS as variantes disponíveis, cada uma com sua explicação e body correspondente. Nunca omita uma variante.
6. Nunca diga que algo "não é possível" ou "não existe" sem ter verificado todos os endpoints do grupo relevante na documentação fornecida.
7. Se após analisar toda a documentação o recurso realmente não existir, informe isso claramente.
8. Substitua as variáveis como {{URL}}, {{token}}, {{contactId}} por descrições claras do que deve ser preenchido.

REGRA DE FORMATO — leia e siga sem repetir estes títulos na resposta:

Quando a pessoa fizer uma pergunta nova, responda APENAS com o comando curl direto e a pergunta final. Não adicione introduções, títulos ou explicações longas. Siga exatamente este padrão de saída:

Aqui está o comando pronto para [descrição breve]:

```
curl -X [MÉTODO] \
  {{URL}}/api/v1/[caminho] \
  -H 'Authorization: Bearer {{token}}' \
  -H 'Content-Type: application/json' \
  -d '{"campo": "{{valor}}"}'
```

[se houver múltiplas variantes do mesmo endpoint com bodies diferentes, repita o bloco acima para cada uma, precedido de "Opção 1 — nome:" e "Opção 2 — nome:". Mostre SOMENTE as variantes que existem na documentação, nunca invente.]

Ficou com alguma dúvida ou quer uma explicação mais detalhada de como configurar isso no Postman?

---

Quando o usuário responder que sim, que quer mais detalhes, ou demonstrar dúvida, aí sim envie a resposta completa:

Primeiro o passo a passo no Postman:
Passo 1: [explique de forma simples, como se a pessoa nunca tivesse usado uma API]
Passo 2: ...
[se houver variantes, use "--- Opção 1: nome ---" e "--- Opção 2: nome ---"]

Depois inclua este texto fixo (copie sem alterar):
Para realizar de forma mais simples, copie e cole esse código no seu Postman, preencha os campos das variáveis e faça a requisição.

JSON para importar via File > Import:
```json
{
  "info": { "name": "[nome]", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json" },
  "item": [
    {
      "name": "[nome]",
      "request": {
        "method": "[MÉTODO]",
        "header": [
          { "key": "Authorization", "value": "Bearer {{token}}", "type": "text" },
          { "key": "Content-Type", "value": "application/json", "type": "text" }
        ],
        "url": { "raw": "{{URL}}/api/v1/...", "host": ["{{URL}}"], "path": ["api","v1","..."] },
        "body": { "mode": "raw", "raw": "{\n  \"campo\": \"{{valor}}\"\n}", "options": { "raw": { "language": "json" } } }
      }
    }
  ]
}
```

INFORMAÇÕES GERAIS DA API DIGISAC:
- URL base: https://SEU-SUBDOMINIO.digisac.co — substitua SEU-SUBDOMINIO pelo subdomínio da sua conta
- Autenticação: todas as requisições precisam do header: Authorization: Bearer SEU_TOKEN
- Para obter o token: POST /api/v1/oauth/token
- O endpoint POST /api/v1/messages tem várias variantes: contato cadastrado (usa contactId), número não cadastrado (usa number + serviceId), sem abrir chamado, via bot, com mídia, entre outros — sempre apresente todas as opções relevantes.
"""


# ---------------------------------------------------------------------------
# Sessões (histórico de conversa)
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict]] = {}
MAX_HISTORY = 6  # mantém últimas 6 mensagens para economizar tokens


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="IA Suporte Digisac",
    description="Assistente de dúvidas sobre a API Digisac, baseado em Groq + LLaMA.",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>IA Suporte Digisac</h1>")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Busca chunks relevantes para a pergunta atual
    relevant_chunks = search_chunks(req.message)
    context = build_context(relevant_chunks)

    # Monta system prompt com contexto dinâmico
    system_prompt = (
        SYSTEM_BASE
        + "\n\nDOCUMENTAÇÃO RELEVANTE PARA ESTA PERGUNTA:\n"
        + "---\n"
        + context
        + "\n---\n"
        + "Responda com base nesses endpoints. Se precisar de mais contexto, peça ao usuário para detalhar."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": req.message})

    # Modelos de fallback usados em ordem quando o principal atinge o limite
    FALLBACK_MODELS = [
        GROQ_MODEL,
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]

    answer = None
    last_error = None

    for model in FALLBACK_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
            )
            answer = completion.choices[0].message.content
            break
        except Exception as e:
            err_str = str(e)
            last_error = err_str
            # Se for rate limit (429), tenta o próximo modelo
            if "429" in err_str or "rate_limit" in err_str:
                continue
            # Qualquer outro erro para imediatamente
            raise HTTPException(status_code=500, detail=f"Erro ao chamar a API Groq: {err_str}")

    if answer is None:
        # Extrai o tempo de espera da mensagem de erro, se disponível
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

    return ChatResponse(response=answer, session_id=session_id)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Sessão encerrada."}


@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL, "chunks": len(API_CHUNKS)}


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
