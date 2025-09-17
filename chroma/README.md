### Main workflow of execution

1. Run C0
    1. Creates the `bookcorpus_sentences` collection with the sentences and 
    embeddings with 0 for the 384 dimensions.
2. Run C1
    1. Reads all the sentences from the previously created collection, 
    and creates a collection called `bookcorpus_sentences_cosine`, with the proper 
    embeddings.
3. Run C2
    1. Copies the `bookcorpus_sentences_cosine` to another called 
    `bookcorpus_sentences_euclidean`, which uses "l2" as the distance function of the 
    embedding space, instead of "cosine".
    2. Using sentences from 
    [our_10_sentences.txt](../data_used/our_10_sentences.txt), compute the top-2 most
    similar sentences among all other sentences in the corresponding collection
    (`bookcorpus_sentences_euclidean` for calculating the Euclidean distance and 
    `bookcorpus_sentences_cosine` for the Cosine distance).

#### Alternative versions

- C0_individual
    - Loads the sentences individually into the collection, committing each insertion
- C1_individual
    - Updates the `bookcorpus_sentences` collection to add the corresponding embedding 
    to each sentence, individually
- C1_update_same_collection
    - Like C1_individual, except it updates the elements of the collection in bulk 
    with two iterations, 5,000 elements at a time
- **Important:** if either C1_individual or C1_update_same_collection are used instead of 
C1, the collection `bookcorpus_sentences_cosine` has to be renamed to `bookcorpus_sentences` 
for C2 to work.
