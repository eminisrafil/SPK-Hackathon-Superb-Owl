import pickle
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from query_gpt import query_gpt

MODEL = "gpt-3.5-turbo"
LOOKAHEAD_STEPS = 1
NUM_SAMPLES = 1


def predict_next_words(root_word, lookahead_steps=2, num_samples=3):
    G = nx.DiGraph()
    G.add_node(root_word)
    queue = deque([(root_word, 0)])  # Each item is a tuple (node, current_depth)
    instructions = "predict next words from a sequence of words:"

    while queue:
        current_node, current_depth = queue.popleft()

        if current_depth < lookahead_steps:
            messages = [dict(role="system", content=instructions), dict(role="user", content=" ".join(current_node))]
            response = query_gpt(model=MODEL, breadth=num_samples, messages=messages)

            for choice in response.choices:
                next_word = choice.message.content
                assert next_word
                child_node = current_node + " " + next_word
                if not G.has_node(child_node):  # Add the node if it doesn't already exist
                    G.add_node(child_node)
                    G.add_edge(current_node, child_node)
                    queue.append((child_node, current_depth + 1))
    return G


def plot_prediction_tree(G):
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True, node_color='b', alpha=0.3)
    plt.show()


def main():
    with open('transcription.pkl', 'rb') as f:
        alternative = pickle.load(f)
        sequence_of_words = [word.word for word in alternative.words]

    G = nx.DiGraph()
    cum_seq = [" ".join(sequence_of_words[:i]) for i in range(1, len(sequence_of_words) + 1)]

    for seq in cum_seq:
        G_i = predict_next_words(seq, lookahead_steps=LOOKAHEAD_STEPS, num_samples=NUM_SAMPLES)
        G = nx.compose(G, G_i)
        plot_prediction_tree(G)


if __name__ == "__main__":
    main()
