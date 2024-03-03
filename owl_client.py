import json
import multiprocessing

import socketio

from client_db import get_db, Utterance

sio = socketio.Client()


# The main function where the connection is established
def run():
    print("running")
    sio.connect('http://127.0.0.1:8000/socket.io', headers={"Authorization": "bearer 6a99985acd0440729b468c1cecd37096"})
    print("connected")
    sio.wait()
    print("waited")


def start_client_thread():
    import threading
    p = threading.Thread(target=run, args=())
    p.start()
    return p


@sio.on("new_utterance")
def on_new_utterance(msg):
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

    db = get_db()
    print("XXXX-new_utterance", msg)
    print("len utterances", len(db.utterances))
    utterance_data = json.loads(msg["utterance"])
    db.utterances.append(Utterance(**utterance_data))
    print("len utterances", len(db.utterances))


# @sio.on("*")
# def on_any_event(event, *args, **kwargs):
#     print("Received unhandled event:", event)
#     print("Arguments:", args)
#     print("Keyword arguments:", kwargs)


if __name__ == '__main__':
    # p = multiprocessing.Process(target=run, args=())
    # p.start()
    p = start_client_thread()
    print("printing db")
    p.join()
