import openai
from joblib import Memory

mem = Memory(".cache", verbose=0)
client = openai.Client()
TEMPERATURE = 0.7


@mem.cache()
def query_gpt(model, messages, max_tokens, n=1):
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
def fetch_embeddings(combined_message):
    embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=combined_message
    )
    return embedding_response
