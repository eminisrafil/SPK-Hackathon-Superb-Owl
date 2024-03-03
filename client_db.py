import multiprocessing


# class DataBase:
#     def __init__(self):
#         self.state = ""
#         self.frames: list = []
#         self.utterances: list['Utterance'] = []


class DataBase:
    _instance = None
    _lock = multiprocessing.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.state = ""
            self.frames = []
            self.utterances = []
            self.initialized = True


class Utterance:
    def __init__(self, start: float, end: float, spoken_at: str, text: str, speaker: str):
        self.start = start
        self.end = end
        self.spoken_at = spoken_at
        self.text = text
        self.speaker = speaker


db = None


def get_db():
    global db
    if db is None:
        db = DataBase()
    return db


if __name__ == "__main__":
    get_db()
