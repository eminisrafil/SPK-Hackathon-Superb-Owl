import pickle

import httpx
import threading
import time
import json

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

from manuel.config import DEEPGRAM_API_KEY

# URL for the realtime streaming audio you would like to transcribe
URL = 'http://stream.live.vc.bbcmedia.co.uk/bbc_world_service'

# Duration for transcription in seconds
TRANSCRIPTION_DURATION = 3


def main():
    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        # Create a websocket connection to Deepgram
        dg_connection = deepgram.listen.live.v("1")

        # Define event handlers
        def on_message(self, result, **kwargs):
            print("number of results: ", len(result.channel.alternatives))
            alternative, = result.channel.alternatives
            with open('../transcription.pkl', 'wb') as f:
                pickle.dump(alternative, f)

            # for alternative in result.channel.alternatives:
            #     for word in alternative.words:
            #         print(f"{word.word} ({word.start} - {word.end}, {word.confidence})")

        def on_metadata(self, metadata, **kwargs):
            print(f"METADATA:\n\n{metadata}\n\n")

        def on_error(self, error, **kwargs):
            print(f"ERROR:\n\n{error}\n\n")

        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # Configure Deepgram options for live transcription
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
        )

        # Start the connection
        dg_connection.start(options)

        # Create a lock and a flag for thread synchronization
        lock_exit = threading.Lock()
        exit = False

        # Define a thread that streams the audio and sends it to Deepgram
        def myThread():
            start_time = time.time()
            with httpx.stream("GET", URL) as r:
                for data in r.iter_bytes():
                    if time.time() - start_time >= TRANSCRIPTION_DURATION:
                        break
                    dg_connection.send(data)

        # Start the thread
        myHttp = threading.Thread(target=myThread)
        myHttp.start()

        # Wait for the specified duration
        myHttp.join(TRANSCRIPTION_DURATION)

        # Set the exit flag to True to stop the thread
        lock_exit.acquire()
        exit = True
        lock_exit.release()

        # Close the connection to Deepgram
        dg_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


if __name__ == "__main__":
    main()
