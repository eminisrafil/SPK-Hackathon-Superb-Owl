import csv
import enum
import threading
import time

import eventlet
from scipy.spatial.distance import cosine

from client_db import fetch_and_process, Utterance
from manuel import server
from manuel.config import VIBE_SMOOTHING_FACTOR, VIBE_PERCENTAGE_THRESHOLD, MODEL_NAME, BEST_MODEL_NAME, \
    COSINE_THRESHOLD
from manuel.query_gpt import query_gpt, fetch_embeddings

"""
companion object {
const val VIBRATIONS_KEY = "vibrations" // 0
const val PROMPT_KEY = "prompt" // “”
const val VIBES_PERCENT_KEY  = "vibesPercent" // INT 0-100
"""


class VIBRATION(enum.IntEnum):
    NO = 0
    ONCE = 1
    CONTINUOUS = 2


def none_cosine(a, b):
    if a is None or b is None:
        return 0
    return cosine(a, b)


def summarize_user_profile(tweets):
    system = """\
You create conversation preferences profiles for users,\
 given a list of tweets you are supposed to describe the user's conversation preferences so that in the future you can suggest conversation topics."""
    element_description = f"""\
You generate , given a list of restaurants and a user's culinary profile.
# Instructions
Review the current information about the user, his personal profile and all other information to best understand he users conversation preferences.
Your profile will be used by another AI to steer future conversations towards a more positive direction.
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": element_description},
        {"role": "user", "content": "\n".join(tweets)}
    ]
    response = query_gpt(model=MODEL_NAME, messages=messages, n=1, max_tokens=50)
    rest_of_message = response.choices[0].message.content
    return rest_of_message


def eval_vibes(text):
    messages = [
        {"role": "system", "content": """\
You are an emotional understanding AI, your job is to evaluate the user vibe level, Respond with a number between 0-100 to show the vibe level of the conversation.
You are part of a meta-agent, you are not directly in contact with the user, your output is limited to numbers between 0-100.
Some examples:
User: I'm loving my life!
You: 100
User: Today is a good day.
You: 50
User: I hate my life, we are all going to die.
You: 0
OUTPUT_FORMAT: int"""
         }, {"role": "user", "content": text}]
    response = query_gpt(model=MODEL_NAME, messages=messages, n=1, max_tokens=2)
    vibe_level = int(response.choices[0].message.content)
    return vibe_level


class State(enum.Enum):
    VIBE_CHECKING = 1
    VIBE_CORRECTING = 2


# out_queue = asyncio.Queue()
out_queue = eventlet.queue.LightQueue()


class VibeCorrector:
    def __init__(self, user_profile):
        self.running_vibe_level = 70
        self.user_vibe_target = 70
        self.state = State.VIBE_CHECKING
        self.user_profile = user_profile
        self.previous_suggestion = (None, None)

    def vibe_check(self, text_so_far):
        current_vibes = eval_vibes(text_so_far)
        print(f"Vibe level: {current_vibes} Text so far: `{text_so_far}`")
        running_vibe_level = VIBE_SMOOTHING_FACTOR * self.running_vibe_level + (
                1 - VIBE_SMOOTHING_FACTOR) * current_vibes

        should_vibe_correct = running_vibe_level + VIBE_PERCENTAGE_THRESHOLD < self.user_vibe_target
        if len(text_so_far.split(" ")) < 5:
            should_vibe_correct = False
        return should_vibe_correct, running_vibe_level

    def step_state_machine(self, text_so_far) -> str:
        suggestion = ""
        if self.state == State.VIBE_CHECKING:
            should_vibe_correct, self.running_vibe_level = self.vibe_check(text_so_far)
            if should_vibe_correct:
                self.state = State.VIBE_CORRECTING
                print(
                    f"Vibe level: {self.running_vibe_level} is not close enough to {self.user_vibe_target} to vibe. Correcting...")
            else:
                print(f"Vibing at {self.running_vibe_level}")
        if self.state == State.VIBE_CORRECTING:
            self.state = State.VIBE_CHECKING
            print("Vibe correction:")
            suggestion = self.steer_conversation(text_so_far)
            suggestion = self.deduplicate_suggestion(suggestion)

            # request post request
            # requests.post("http://localhost:5000/api/v1/ai/steer_conversation", json={"text": out, "vibration": VIBRATION.ONCE})
            print("Vibe correction: ", suggestion)
        return suggestion

    def deduplicate_suggestion(self, suggestion):
        # Avoid suggesting two similar topics
        current_embedding = fetch_embeddings(suggestion)
        print(current_embedding.sum(), "Suggestion", suggestion)
        previous_suggestion, previous_embedding = self.previous_suggestion
        similarity = none_cosine(current_embedding, previous_embedding)
        print("Similarity: ", similarity, end=" ")
        if similarity > COSINE_THRESHOLD:
            print("keeping the previous suggestion")
            suggestion = self.previous_suggestion[0]
        else:
            print("updating a suggestion")
            self.previous_suggestion = (suggestion, current_embedding)
        return suggestion

    def steer_conversation(self, text):
        print("Steering conversation")
        messages = [
            {"role": "system", "content": """\
You are inside a graphical display, You are part of a meta-agent, you can communicate with the user through messages displayed in a small screen to the user,
Your goal is to steer the conversation towards a more positive direction.
the user won't be able to answer but will read your message.
Utilize your knowledge of the user's interests and the current conversation to suggest a new topic.
"""},
            # # Analyze:
            # Review the current information about the user, their personal profile and all other information to find how to best help the user.
            # Reason about the current conversation and the user's interests to find a new topic to suggest.
            # Integrate each piece of information about the and think about its relevance to your possible actions.
            # # Hypothesize:
            # Hypothesize about the user's reaction to the new topic and how likely they will accept it.
            # # Criticize:
            # Tools come with a cognitive cost, we want the user to continue using the app for a long time and suggesting a tool when not strictly necessary will risk the user uninstalling the app.
            # After reasoning about the tools, criticize the idea of invoking a tool, think about the cognitive cost and the potential negative effects of invoking a tool.
            # # Recommend:
            # After criticizing the idea of invoking a tool, recommend the best course of action to the next agent.
            # """},
            {"role": "user", "content": f"""\
User profile:
{self.user_profile}
Conversation so far:
{text}
"""},
        ]
        response = query_gpt(model=MODEL_NAME, messages=messages, n=1, max_tokens=50)
        rest_of_message = response.choices[0].message.content
        messages = [
            {"role": "system", "content": """\
Your job is to summarize a from another agent whose goal is to steer the conversation towards a more positive direction.
You will receive a longer message and will have extract a single actionable message to guide the conversation,
Your message will be displayed in a small display to the user, only few words will fit.
Be concise, pick the best few word to prompt the user to steer the conversation.
Write an actionable practical message not a full sentence. the user has no time to think, suggest directly"""},
            {"role": "user", "content": f"""\
Conversation so far:
{text}
Suggested interjection message:
{rest_of_message}
"""},
        ]
        response = query_gpt(model=BEST_MODEL_NAME, messages=messages, n=1, max_tokens=5)
        brief_message = response.choices[0].message.content
        print("Steering conversation: ", brief_message)
        return brief_message


def get_profile():
    tweets = []
    with open('TwExportly_EthanSutin_tweets_2024_03_02.csv', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row_dict = dict(zip(header, row))
            print(row_dict['type'])
            if row_dict['type'].lower() != 'retweet':
                tweets.append(row_dict['text'])
    user_profile = summarize_user_profile(tweets)
    return user_profile


def get_corrector():
    user_profile = get_profile()
    vibe_corrector = VibeCorrector(user_profile)
    return vibe_corrector


def get_cum_seq():
    return fetch_and_process(Utterance)


def offline_replay():
    server.set_queue(out_queue)
    threading.Thread(target=server.run_server).start()

    vibe_corrector = get_corrector()
    while True:
        for text_so_far in get_cum_seq():
            steering_prompt = vibe_corrector.step_state_machine(text_so_far)
            if steering_prompt:
                out_queue.put_nowait((VIBRATION.ONCE, steering_prompt, vibe_corrector.running_vibe_level))
                print(steering_prompt)
            else:
                out_queue.put_nowait((VIBRATION.NO, "", vibe_corrector.running_vibe_level))
            time.sleep(5)
        print("Done")


if __name__ == "__main__":
    offline_replay()
