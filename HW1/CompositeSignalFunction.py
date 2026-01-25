from SignalProcessor import SignalProcessor

class CompositeSignalFunction(SignalProcessor):
    def __init__(self, inputs):
        super().__init__([])
        if not inputs:
            raise ValueError("CompositeSignalFunction requires at least one input signal.")
        self.inputs = list(inputs)

    def __call__(self, duration):
        if duration < 0:
            raise ValueError("duration must be non-negative")

        # Ensure each input signal has data for the requested duration
        for signal in self.inputs:
            if len(signal) != duration:
                signal(duration)

        self.data = []
        for idx in range(duration):
            sample_total = 0
            for signal in self.inputs:
                sample_total += signal.data[idx]
            self.data.append(sample_total)
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration

        value = self.data[self.idx]
        self.idx += 1
        return value

    def __eq__(self, other):
        if not hasattr(other, "data"):
            return NotImplemented

        if len(other) != len(self.data):
            raise ValueError("Two signals are not equal in length!")

        count = 0
        for i in range(len(self.data)):
            if abs(self.data[i] - other.data[i]) < 0.01:
                count += 1
        return count
