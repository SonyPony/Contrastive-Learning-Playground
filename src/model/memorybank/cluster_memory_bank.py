class ClusterMemoryBank:
    def __init__(self):
        self._memory = dict()

    def __setitem__(self, key: int, value):
        self._memory[key] = value

    def __getitem__(self, key: int):
        return self._memory.get(key, default=None)
