import collections
import json
import threading
from multiprocessing import Queue

import socketio

from manuel import server
from vibe_corrector import get_corrector, VIBRATION

vibe_corrector = get_corrector()
in_queue = Queue(maxsize=10)
out_queue = Queue(maxsize=10)

sio_data_in = socketio.Client()


def produce_data():
    sio_data_in.connect('http://127.0.0.1:8000/socket.io', headers={"Authorization": "bearer 6a99985acd0440729b468c1cecd37096"})
    sio_data_in.wait()


@sio_data_in.on("new_utterance")
def on_new_utterance(msg):
    utterance_data = json.loads(msg["utterance"])
    in_queue.put(utterance_data["text"])
    print("len(in_queue)", len(in_queue._buffer))


def process():
    data_in_thread = threading.Thread(target=produce_data)
    data_in_thread.start()

    data_out_thread = threading.Thread(target=server.forward_data)
    data_out_thread.start()

    server.set_queue(out_queue)
    cum_text = collections.deque(maxlen=2)

    try:
        while True:
            print("Waiting for data")
            next_word = in_queue.get()
            print("Got data", next_word)
            cum_text.append(next_word)
            text_so_far = " ".join(cum_text)
            steering_prompt = vibe_corrector.step_state_machine(text_so_far)
            if steering_prompt:
                out_queue.put_nowait((VIBRATION.ONCE, steering_prompt, vibe_corrector.running_vibe_level))
            else:
                out_queue.put_nowait((VIBRATION.NO, "", vibe_corrector.running_vibe_level))
            print("len(out_queue)", len(out_queue._buffer))
    finally:
        data_in_thread.join()
        data_out_thread.join()


if __name__ == '__main__':
    process()
