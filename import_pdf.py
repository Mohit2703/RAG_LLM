import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
import random
import re
from sentence_transformers import SentenceTransformer


def text_formatter(text: str) -> str:
    """Perform minor formatting on extracted text"""
    cleaned_text = text.replace("\n", " ").strip()

    """more text formatting"""
    return cleaned_text


def open_read_pdf(pdf_path: str) -> list[dict]:
    """Open and read PDF file"""
    doc = fitz.open(pdf_path)
    pages_and_text = []

    for page_num, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_text.append(
            {
                "page": page_num - 3,
                "page_char_count": len(text),
                "page_words_count": len(text.split(" ")),
                "page_sentences_count": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
            }
        )

    return pages_and_text


""" Spliting the pages into sentences  
1. by spliting of ". "
2. by using nlp library spacy and nltk
"""


def split_sentences(pages_and_text: list[dict]) -> list[dict]:
    nlp = English()
    nlp.add_pipe("sentencizer")
    for page in tqdm(pages_and_text):
        sentences = nlp(page["text"])
        page["sentences"] = [str(sent.text) for sent in sentences.sents]
        page["page_sentences_count_spacy"] = len(page["sentences"])

    return pages_and_text


""" Chunking the sentences together 
will split into group of 10 sentences 
have langchain framework to do this
for now continue with python"""


def split_sentences_list(pages_and_text: list[dict], chunk_size=5) -> list[dict]:
    for page in tqdm(pages_and_text):
        page["sentence_chunks"] = [
            page["sentences"][i : i + chunk_size]
            for i in range(0, len(page["sentences"]), chunk_size)
        ]
        page["sentence_chunks_count"] = len(page["sentence_chunks"])

    return pages_and_text


""" 
# Spliting each chunk into its own item 
We would like to embed each chunk into its own numberical representation
"""


def split_sentences_chunk(pages_and_text: list[dict]) -> list[dict]:
    pages_and_chunk = []

    for page in tqdm(pages_and_text):
        for sentence_chunk in page["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page"] = page["page"]
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            chunk_dict["chunk"] = re.sub(r"\.([A-Z])", r". \1", joined_sentence_chunk)
            chunk_dict["chunk_char_count"] = len(chunk_dict["chunk"])
            chunk_dict["chunk_words_count"] = len(chunk_dict["chunk"].split(" "))
            chunk_dict["chunk_token_count"] = len(chunk_dict["chunk"]) / 4
            pages_and_chunk.append(chunk_dict)
    return pages_and_chunk


""" remove chunks with less than 30 token count """


def remove_small_chunks(pages_and_text: list[dict]) -> list[dict]:
    return [page for page in pages_and_text if page["chunk_token_count"] > 30]


""" 
Embedding our text chunks
- Turn text chunks into numbers, specially embeddings
"""


def embed_text_chunks(pages_and_chunks: list[dict]) -> list[dict]:
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    for page in tqdm(pages_and_chunks):
        sentence_embeddings = embedding_model.encode(page["chunk"])
        page["embedding"] = sentence_embeddings

    return pages_and_chunks


""" 
 Batch embeddding chunks
"""


def batch_embed_text_chunks(text_chunks: list[str], batch_size=30) -> list[dict]:
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    text_chunk_embeddings = embedding_model.encode(
        text_chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    return text_chunk_embeddings


if __name__ == "__main__":
    pdf_path = "./GID.pdf"
    pages_and_text = open_read_pdf(pdf_path)
    pages_and_text = split_sentences(pages_and_text)
    pages_and_text = split_sentences_list(pages_and_text)
    print(len(pages_and_text))

    pages_and_chunks = split_sentences_chunk(pages_and_text)
    pages_and_chunks = remove_small_chunks(pages_and_chunks)

    embedded_chunks = embed_text_chunks(pages_and_chunks)

    text_chunks = [chunk["chunk"] for chunk in embedded_chunks]

    text_chunk_embeddings = batch_embed_text_chunks(text_chunks)

    df = pd.DataFrame(embedded_chunks)

    df.to_csv("embedded_chunks.csv", index=False)
    # print(len(text_chunks))
