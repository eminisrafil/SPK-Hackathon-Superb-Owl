import eventlet
import socketio

q = None


def set_queue(queue):
    global q
    q = queue


sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})


@sio.event
def connect(sid, environ):
    print('connect ', sid)
    sio.start_background_task(send_vibe, sid, None)


@sio.event
def send_vibe(sid, data):
    global q
    while q is None:
        sio.sleep(1)

    while True:
        print("\t GETTING")
        (vi, prompt, perc) = q.get()
        print(f'SENDING VIBE {vi}:{prompt}:{perc}')
        sio.emit('vibes', {
            "vibes": vi,
            "prompt": prompt,
            "vibesPercent": int(perc),
        }, room=sid)
        print("\t SENT")
        sio.sleep(1)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


def run_server():
    eventlet.wsgi.server(eventlet.listen(('', 8765)), app)


if __name__ == '__main__':
    run_server()
