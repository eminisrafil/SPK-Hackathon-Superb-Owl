import pickle

import numpy as np
from scipy.spatial.distance import cosine

from query_gpt import query_gpt, fetch_embeddings

# Global variables to store the last message embedding and response
last_message_embedding = None
last_response = None


def check_if_same(combined_message):
    global last_message_embedding, last_response

    # Embedding the combined message
    embedding_response = fetch_embeddings(combined_message)
    current_embedding = np.array(embedding_response.data[0].embedding)

    # Calculate similarity if last_message_embedding exists
    if last_message_embedding is not None:
        similarity = 1 - cosine(last_message_embedding, current_embedding)
        if similarity > 0.9:
            # If similar, return True and the similarity score
            return True, similarity, last_response

    # Update the last message embedding and return False if not similar
    last_message_embedding = current_embedding
    return False, 0, None


def process4(text):
    global last_response

    # New OpenAI call to assess completeness of the user message
    end_likelihood = completeness(text)
    print(end_likelihood)

    # If the likelihood is 90 or above, consider the message complete and skip to similarity check
    if end_likelihood >= 90:
        combined_message = text
    else:
        # Predict the ending of the user message if likelihood is below 90
        combined_message = autocomplete(text)

    # Check if the current message is similar to the last one
    is_similar, similarity, saved_response = check_if_same(combined_message)
    if is_similar:
        # Use the last response if similar
        return {'response': saved_response, 'rest_of_message': combined_message, 'similarity': similarity, 'end_likelihood': end_likelihood}

    # Generate a response if not similar or message is considered complete
    output = precompute_response(combined_message)

    # Update the last response
    last_response = output
    return {'response': output, 'rest_of_message': combined_message, 'end_likelihood': end_likelihood}


def precompute_response(combined_message):
    messages = [
        {"role": "system", "content": """Generate a response to the user message"""},
        {"role": "user", "content": combined_message}
    ]
    response = query_gpt(model="gpt-3.5-turbo", messages=messages, n=1, max_tokens=50)
    output = response.choices[0].message.content
    return output


def autocomplete(text):
    messages = [
        {"role": "system", "content": """\
You are an AI designed to predict the end of a user message. Generate the expected ending of this user message. Do not include a response to the user message. """},
        {"role": "user", "content": text}
    ]
    response = query_gpt(model="gpt-3.5-turbo", messages=messages, n=1, max_tokens=50)
    rest_of_message = response.choices[0].message.content
    combined_message = f"{text} {rest_of_message}"
    return combined_message


def completeness(text):
    messages = [
        {"role": "system", "content": "Respond with a number between 0-100 to show likelihood that the user message is complete and can be responded to."},
        {"role": "user", "content": "Hi."},
        {"role": "user", "content": "100"},
        {"role": "user", "content": "What is your "},
        {"role": "user", "content": "0"},
        {"role": "user", "content": text}
    ]
    response = query_gpt(model="gpt-3.5-turbo", messages=messages, n=1, max_tokens=2)
    end_likelihood = int(response.choices[0].message.content)
    return end_likelihood


def main():
    with open('transcription.pkl', 'rb') as f:
        alternative = pickle.load(f)
        sequence_of_words = [word.word for word in alternative.words]

    cum_seq = [" ".join(sequence_of_words[:i]) for i in range(1, len(sequence_of_words) + 1)]

    for text_so_far in cum_seq:
        response = process4(text_so_far)
        print(response)


if __name__ == "__main__":
    main()
