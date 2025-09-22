CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL
);

CREATE TABLE embeddings_pgvector (
    id SERIAL PRIMARY KEY REFERENCES sentences(id),
    embedding VECTOR(384) NOT NULL
);

-- HNSW index for cosine distance
CREATE INDEX ON embeddings_pgvector USING hnsw (embedding vector_cosine_ops);

-- HNSW index for L2/Euclidean distance
CREATE INDEX ON embeddings_pgvector USING hnsw (embedding vector_l2_ops);
