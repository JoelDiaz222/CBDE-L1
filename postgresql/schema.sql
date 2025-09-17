CREATE TABLE sentences (
    id SERIAL PRIMARY KEY,
    sentence TEXT NOT NULL
);

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY REFERENCES sentences(id),
    embedding FLOAT8[] NOT NULL
);
