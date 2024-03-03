import json
import logging
import multiprocessing

import socketio

from client_db import Database, Context, create_context, create_utterance, Utterance, get_active_context

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

db = Database().get_session()
active_context = None

# Create an instance of the Async Socket.IO Client
sio = socketio.Client()


# Event handler for when the client connects to the server
@sio.event
def connect():
    global active_context
    print('Connected to the server.')
    context = Context()
    active_context = create_context(db, context)


# Event handler for when the client disconnects from the server
@sio.event
def disconnect():
    del db


@sio.on("new_utterance")
def on_new_utterance(msg):
    utterance = json.loads(msg["utterance"])
    utterance = Utterance(
        context=active_context,
        **utterance
    )
    create_utterance(db, utterance)


# The main function where the connection is established
def run():
    sio.connect('http://127.0.0.1:8000/socket.io', headers={"Authorization": "bearer your_own_secret_token"})
    sio.wait()


def start_client():
    p = multiprocessing.Process(target=run, args=())
    p.start()
    return p


def fetch_and_process(model, buffer_size=100):
    with Database() as db:
        ctx: Context = get_active_context(db)
        query = db.query(model).filter(model.context_id == ctx.id)
        buffer = []

        for instance in query.yield_per(buffer_size):
            buffer.append(instance)
            if len(buffer) >= buffer_size:
                # Process buffer here, e.g., yield to another process or perform some operation
                print(f"Processing {len(buffer)} items from {model.__tablename__}")
                buffer.clear()
        if buffer:
            # Process any remaining items in the buffer
            print(f"Processing remaining {len(buffer)} items from {model.__tablename__}")
            buffer.clear()
        return buffer


if __name__ == '__main__':
    # start owl client in a seprate process
    p = start_client()
    for b in fetch_and_process(Utterance):
        print(b)
    # try to join porcess
    p.join()
