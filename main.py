import os
from fastapi import FastAPI
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

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    # 3.1 Embedding ερωτήματος
    q_emb = client.embeddings.create(
        input=req.question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # 3.2 Αναζήτηση στο Pinecone (top‑k = 30)
    res = index.query(vector=q_emb, top_k=30, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in res["matches"]])

    # 3.3 Κλήση ChatGPT
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