from SineWaveFunction import SineWaveFunction
from SquareWaveFunction import SquareWaveFunction

def main():
    duration_short = 10
    duration_mismatch = 6

    print("Initializing waves")
    # Sine
    sine1 = SineWaveFunction(amplitude=1.0, frequency=0.10)
    sine1_copy = SineWaveFunction(amplitude=1.0, frequency=0.10)
    sine2 = SineWaveFunction(amplitude=2.0, frequency=0.10)
    sine3 = SineWaveFunction(amplitude=1.0, frequency=0.20)

    # Square
    sq1 = SquareWaveFunction(amplitude=1.0, frequency=0.10)
    sq2 = SquareWaveFunction(amplitude=2.0, frequency=0.10)
    sq3 = SquareWaveFunction(amplitude=1.0, frequency=0.20)

    print("__call__ test")
    print("Sine Waves")
    sine1(duration_short)
    sine1_copy(duration_short)
    sine2(duration_mismatch)
    sine3(duration_short)

    print("Square Waves")
    sq1(duration_short)
    sq2(duration_mismatch)
    sq3(duration_short)

    print("__iter__/__next__ test")
    print("Sine test")
    for val in sine1:
        print(val)

    print("Square test")
    for val in sq1:
        print(val)

    print("__len__ test")
    print("Sine test")
    print(len(sine1))
    print(len(sine2))
    print(len(sine3))

    print("Square test")
    print(len(sq1))
    print(len(sq2))
    print(len(sq3))

    print("__eq__ test")
    print("Sine1 vs Sq1: 0")
    print(sine1 == sq1)

    print("Sine1 vs Sine1_copy: 10")
    print(sine1 == sine1_copy)

    print("Sine1 vs Sine3: 2")
    print(sine1 == sine3)

    print("Sq1 vs Sq2: 5")
    print(sq1 == sq3)

    print("Compare 0.01 Tolerance")
    test1 = SineWaveFunction(amplitude=1.0, frequency=0.10)
    test2 = SineWaveFunction(amplitude=1.01, frequency=0.10)
    test1(2)
    test2(2)
    print(test1 == test2)


    print("Sine1 vs Sine2: duration mismatch")
    print(sine1 == sine2)

if __name__ == "__main__":
    main()
