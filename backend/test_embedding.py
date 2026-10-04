from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

texts = [
    "Information retrieval finds relevant documents for a user's query.",
    "PostgreSQL is a relational database management system.",
    "Neural networks are widely used in machine learning.",
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Model:", MODEL_NAME)
print("Embedding shape:", embeddings.shape)
print("Dimension:", embeddings.shape[1])

for text, embedding in zip(texts, embeddings):
    print("\nText:", text)
    print("First 10 dimensions:", embedding[:10])
