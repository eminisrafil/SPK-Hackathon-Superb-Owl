import asyncio
import json
import logging

import socketio

from client_db import Database, Context, create_context, create_utterance, Utterance

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

db = Database().get_session()
active_context = None

# Create an instance of the Async Socket.IO Client
sio = socketio.AsyncClient()


# Event handler for when the client connects to the server
@sio.event
async def connect():
    global active_context
    print('Connected to the server.')
    context = Context()
    active_context = create_context(db, context)


# Event handler for when the client disconnects from the server
@sio.event
async def disconnect():
    del db


@sio.on("new_utterance")
async def on_new_utterance(msg):
    if not msg:
        return
    utterance = json.loads(msg["utterance"])
    utterance = Utterance(
        context=active_context,
        **utterance
    )
    create_utterance(db, utterance)


# @sio.on("new_human")
# async def on_new_human(msg):
#     human = json.loads(msg["human"])
#     Human(
#         context=active_context,
#         **human
#     )
#     create_human(db, human)


# The main function where the connection is established
async def run():
    await sio.connect('http://127.0.0.1:8000/', headers={"Authorization": "bearer your_own_secret_token"})
    await sio.wait()



if __name__ == '__main__':
    asyncio.run(run())
