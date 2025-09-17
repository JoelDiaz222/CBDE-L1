CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sentences_pgvector (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL,
    embedding vector(384)
);

-- HNSW index for cosine distance
CREATE INDEX ON sentences_pgvector USING hnsw (embedding vector_cosine_ops);

-- HNSW index for L2/Euclidean distance
CREATE INDEX ON sentences_pgvector USING hnsw (embedding vector_l2_ops);
