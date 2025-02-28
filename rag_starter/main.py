import json
import openai

from rag_starter.config import (
    OPENAI_API_KEY,
    INPUT_FILE_PATH,
    EMBEDDINGS_FILE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    CHAT_MODEL,
    CHAT_INSTRUCTIONS,
)

openai.api_key = OPENAI_API_KEY


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into chunks with a specified size and overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def create_embedding_for_text(text):
    """
    Creates an embedding for the provided text using OpenAI.
    """
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data


def similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two embeddings.
    """
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = sum(a**2 for a in embedding1)**0.5
    magnitude2 = sum(a**2 for a in embedding2)**0.5
    return dot_product / (magnitude1 * magnitude2)


def find_similar(embeddings, target_embedding):
    """
    Finds the five most similar chunks compared to the target embedding.
    """
    results = []
    target_vector = target_embedding[0].embedding
    for emb in embeddings:
        sim = similarity(emb["embedding"], target_vector)
        results.append((emb["text"], sim))
    results.sort(key=lambda x: x[1], reverse=True)
    print(results[:5])
    return results[:5]


def process_embeddings():
    """
    Reads the input text, splits it into chunks, creates embeddings,
    and writes the embeddings to a JSON file.
    """
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text_into_chunks(text)
    embeddings = []

    if input("Overwrite old embeddings? (y/n): ").strip().lower() == "y":
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}")
            emb_data = create_embedding_for_text(chunk)
            for em in emb_data:
                embeddings.append({"text": chunk, "embedding": em.embedding})
        with open(EMBEDDINGS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)
    else:
        print("Skipping embedding generation.")


def interactive_chat():
    """
    Loads the embeddings and starts an interactive chat loop.
    """
    with open(EMBEDDINGS_FILE_PATH, "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    messages = [{"role": "system", "content": CHAT_INSTRUCTIONS}]
    while True:
        target_text = input("> ")
        target_embedding = create_embedding_for_text(target_text)
        similar_results = find_similar(embeddings, target_embedding)
        messages.append({
            "role":
            "user",
            "content":
            target_text + "\nRELEVANT PASSAGES FROM DOCUMENTS: " +
            json.dumps(similar_results)
        })
        completion = openai.chat.completions.create(model=CHAT_MODEL,
                                                    messages=messages)
        assistant_reply = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})
        print("\n" + assistant_reply + "\n")


def main():
    process_embeddings()
    interactive_chat()


if __name__ == "__main__":
    main()
