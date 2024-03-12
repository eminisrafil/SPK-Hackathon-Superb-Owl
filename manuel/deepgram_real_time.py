import asyncio
import collections
import json

import pyaudio
import websockets

from vibe_corrector import VibeCorrector

DEEPGRAM_API_KEY = 'b1fd9c45ab6a90f205b2cd887701c854f305ff2c'

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000

audio_queue = asyncio.Queue()
vibe_corrector = VibeCorrector(user_profile=None)

started = False


def callback(input_data, frame_count, time_info, status_flags):
    audio_queue.put_nowait(input_data)
    return (input_data, pyaudio.paContinue)


async def microphone():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback
    )

    stream.start_stream()

    while stream.is_active():
        await asyncio.sleep(0.1)

    stream.stop_stream()
    stream.close()


async def process():
    extra_headers = {
        'Authorization': 'token ' + DEEPGRAM_API_KEY
    }

    async with websockets.connect('wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1', extra_headers=extra_headers) as ws:
        async def sender(ws):  # sends audio to websocket
            try:
                while True:
                    data = await audio_queue.get()
                    await ws.send(data)
            except Exception as e:
                print('Error while sending: ', str(e))
                raise

        async def receiver(ws):
            global started
            if not started:
                print("Started")
                started = True

            words_buffer = collections.deque(maxlen=10)

            async for msg in ws:
                msg = json.loads(msg)
                if 'channel' not in msg:
                    print("Received message: ", msg)
                    continue
                transcript = msg['channel']['alternatives'][0]['transcript']

                if transcript:
                    print(f'Transcript = {transcript}')
                    words_buffer.extend(transcript.split())
                    vibe_corrector.step_state_machine(" ".join(words_buffer))

        await asyncio.gather(sender(ws), receiver(ws))


async def run():
    await asyncio.gather(microphone(), process())


if __name__ == '__main__':
    asyncio.run(run())
