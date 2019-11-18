import sounddevice as sd
from scipy.io.wavfile import write

from pydub import AudioSegment

fs = 16000  # Sample rate


if __name__ == "__main__":
    seconds = int(input("how many seconds should I record? (must be greater than 2): "))
    while seconds < 2:
        seconds = int(input("how many seconds should I record? (must be greater than 2): "))

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write('demo.wav', fs, myrecording, )  # Save as WAV file

    print("done recording, saving recording")
    audio_sample = AudioSegment.from_wav('demo.wav')

    start = 0
    count = 0
    while start + 2 < seconds:
        sample_segment = audio_sample[start * 1000: (start + 2) * 1000]
        sample_segment.export(f"demo/{count}.wav", format="wav", )
        start += 0.25
        count += 1
