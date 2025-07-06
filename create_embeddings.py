import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX", "cabin-prices")
cloud = os.getenv("PINECONE_CLOUD", "gcp")
region = os.getenv("PINECONE_REGION", "us-east1-gcp")

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

index = pc.Index(index_name)

df = pd.read_csv("cabin_prices_clean.csv")

batch = []
batch_size = 100

for i, row in df.iterrows():
    text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
    emb = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding
    batch.append({"id": str(i), "values": emb, "metadata": {"text": text}})
    
    # Όταν το batch φτάσει το batch_size, κάνουμε upsert
    if len(batch) == batch_size:
        index.upsert(vectors=batch)
        print(f"Upserted {i + 1}/{len(df)} vectors...")
        batch = []

# Upsert ό,τι έμεινε
if batch:
    index.upsert(vectors=batch)
    print(f"Upserted {len(df)}/{len(df)} vectors...")

print("Embeddings uploaded!")