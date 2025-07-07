import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# 1) OpenAI client
client = OpenAI()

# 2) Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "cabin-prices")
index = pc.Index(index_name)

# 3) FastAPI app
app = FastAPI(title="Cabin Assistant")

# 4) Serve static files (PDF + chat) από τον φάκελο "static"
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# 5) Chat endpoint
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    # 5.1 Embedding ερωτήματος
    q_emb = client.embeddings.create(
        input=req.question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # 5.2 Αναζήτηση στο Pinecone (top‑k = 30)
    res = index.query(vector=q_emb, top_k=30, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in res["matches"]])

    # 5.3 Κλήση ChatGPT
    system_prompt = (
        "Είσαι βοηθός πωλήσεων για καμπίνες μπάνιου. "
        "Χρησιμοποίησε μόνο τις παρακάτω πληροφορίες για να απαντήσεις.\n\n"
        + context +
        "\n\nΑν βρεις περισσότερες από μία τιμές, εμφάνισέ τες όλες."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.question}
        ]
    )
    answer = completion.choices[0].message.content.strip()
    return {"answer": answer}