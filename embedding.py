from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

# create a list of sentences

sentences = ["This is an example sentence", "Each sentence is converted to a numerical representation", "This is done using embeddings"]

# create a list of embedding
sentence_embeddings = embedding_model.encode(sentences)
embedding_dict = dict(zip(sentences, sentence_embeddings))

# print the embeddings
for sentence, embedding in embedding_dict.items():
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}")
    print("\n")

print(embedding.shape)