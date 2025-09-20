### Main workflow of execution

1. Create and connect to a locally hosted PostgreSQL database named `cbde`.
2. Run the DDL queries in [schema.sql](schema.sql).
   1. Creates the `sentences_pgvector` relation.
3. Run G0.
    1. Loads all the sentences from the 
    [bookcorpus_sentences.txt](../data_used/bookcorpus_sentences.txt) into the 
    `sentences_pgvector` table.
4. Run G1.
    1. Reads all the sentences from the previously populated table, generates 
    embeddings from them using the `all-MiniLM-L6-v2` and stores them in the `embeddings` 
    column.
5. Run G2.
    1. Using sentences from 
    [our_10_sentences.txt](../data_used/our_10_sentences.txt), computes the top-2 most
    similar sentences among all other sentences in the `sentences_pgvector` table using the 
    corresponding embeddings, calculating the Euclidean distance and the Cosine distance.

#### Alternative versions

- G0_two_tables, G1_two_tables and G2_two_tables contain the same business 
logic as G0, G1 and G2, but use two tables instead of one, separating embeddings from 
sentences.
- To run them, the appropriate tables have to be created 
([schema_two_tables.sql](alternative_versions/schema_two_tables.sql)).
