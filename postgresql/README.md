### Main workflow of execution

1. Create and connect to a locally hosted PostgreSQL database named `cbde`.
2. Run the DDL queries in [schema.sql](schema.sql).
   1. Creates the `sentences` and `embeddings` relations.
3. Run P0.
    1. Loads all the sentences from the 
    [bookcorpus_sentences.txt](../data_used/bookcorpus_sentences.txt) into the 
    `sentences` table.
4. Run P1.
    1. Reads all the sentences from the previously populated table, generates 
    embeddings from them using the `all-MiniLM-L6-v2` and stores them in the `embeddings` 
    table.
5. Run P2.
    1. Using sentences from 
    [our_10_sentences.txt](../data_used/our_10_sentences.txt), computes the top-2 most
    similar sentences among all other sentences in the `sentences` table using the 
    corresponding embeddings from the `embeddings` table, calculating the Euclidean distance 
    and the Cosine distance.

#### Alternative versions

- P0_individual.
    - Loads the sentences individually into the corresponding table, committing each insertion
- P0_insert_10k_page_size.
    - Loads the all 10k sentences at once, using a 10,000 element page size for the operation
- P1_individual.
    - Inserts the corresponding embedding for each sentence into the `embeddings` table, 
    commiting each insertion.
- P2_different_queries.
    - Uses different queries than P2 to do the same calculations, that are slightly slower on 
    average considering both distance types.
