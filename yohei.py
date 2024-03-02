import pickle

from scipy.spatial.distance import cosine

from query_gpt import query_gpt, fetch_embeddings

COSINE_THRESHOLD = 0.9


def none_cosine(a, b):
    if a is None or b is None:
        return 1
    return 1 - cosine(a, b)


def precompute_response(combined_message):
    messages = [
        {"role": "system", "content": """Generate a response to the user message"""},
        {"role": "user", "content": combined_message}
    ]
    response = query_gpt(model="gpt-3.5-turbo", messages=messages, n=1, max_tokens=50)
    output = response.choices[0].message.content
    print(f"Precomputed: {output}", end=" ")
    return output


def autocomplete(text):
    messages = [
        {"role": "system", "content": """\
You are an AI designed to predict the end of a user message. Generate the expected ending of this user message. Do not include a response to the user message. """},
        {"role": "user", "content": text}
    ]
    response = query_gpt(model="gpt-3.5-turbo", messages=messages, n=1, max_tokens=50)
    rest_of_message = response.choices[0].message.content
    print(f"Predicted: {rest_of_message}", end=" ")
    return rest_of_message


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
    print(f"Completeness: {end_likelihood}", end=" ")
    return end_likelihood


def main():
    with open('transcription.pkl', 'rb') as f:
        alternative = pickle.load(f)
        sequence_of_words = [word.word for word in alternative.words]

    cum_seq = [" ".join(sequence_of_words[:i]) for i in range(1, len(sequence_of_words) + 1)]
    previous_embedding = None
    previous_prediction = ""

    for text_so_far in cum_seq:
        print(f"Text so far: `text_so_far`", end=" ")
        if completeness(text_so_far) >= 90:
            previous_prediction = ""
            previous_embedding = None
            continue

        predicted_text = autocomplete(text_so_far)

        predicted_message = f"{text_so_far} {predicted_text}"
        current_embedding = fetch_embeddings(predicted_message)

        similarity = 1 - none_cosine(current_embedding, previous_embedding)
        if similarity > COSINE_THRESHOLD:
            print(previous_prediction)
        else:
            predicted_text = precompute_response(predicted_text)  # async

        previous_embedding = current_embedding
        previous_prediction = predicted_text
        print()


if __name__ == "__main__":
    main()
