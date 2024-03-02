import numpy as np
import openai
from joblib import Memory

mem = Memory(".cache", verbose=0)
client = openai.Client()
TEMPERATURE = 0.7


@mem.cache()
def _query_gpt(max_tokens, messages, model, n):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=n,
        stop=None,
        temperature=TEMPERATURE,
    )
    return response


@mem.cache()
def _fetch_embeddings(combined_message):
    embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=combined_message
    )
    emb = np.array(embedding_response.data[0].embedding)
    return emb


def fetch_embeddings(combined_message):
    # print(f"fetching embeddings for {combined_message}")
    emb = _fetch_embeddings(combined_message)
    return emb


def query_gpt(model, messages, max_tokens, n=1):
    # text = " ".join([m["content"] for m in messages])
    # print(f"completions: {model}, {text}")
    response = _query_gpt(max_tokens, messages, model, n)
    return response
