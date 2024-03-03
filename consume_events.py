import socketio

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')


@sio.on('*')
def my_message(*args, **kwargs):
    print('message received:', args, kwargs)


@sio.event
def disconnect():
    print('disconnected from server')


def listen_forever():
    while True:
        try:
            sio.connect('http://localhost:8765')
            sio.wait()
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    listen_forever()
