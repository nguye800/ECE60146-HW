from SignalProcessor import SignalProcessor
import math

class SquareWaveFunction(SignalProcessor):
    def __init__(self, amplitude, frequency):
        # Your implementation here
        super().__init__([])
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, duration):
        self.data = []
        for n in range(duration):
            if math.sin(2 * math.pi * self.frequency * n) >= 0:
                self.data.append(self.amplitude)
            else:
                self.data.append(-self.amplitude)
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