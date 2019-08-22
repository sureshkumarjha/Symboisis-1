#Python 2.x program to transcribe an Audio file 
import speech_recognition as sr 
import pyaudio
import wave

from googletrans import Translator
translator = Translator()

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")
 
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

AUDIO_FILE = (r"file.wav") 

# use the audio file as the audio source 

r = sr.Recognizer() 

with sr.AudioFile(AUDIO_FILE) as source: 
	#reads the audio file. Here we use record instead of 
	#listen 
	audio = r.record(source) 

try: 
	print("The audio file contains: " + r.recognize_google(audio)) 


except sr.UnknownValueError: 
	print("Google Speech Recognition could not understand audio") 

except sr.RequestError as e: 
	print(e) 
print(r.recognize_google(audio),)
s=r.recognize_google(audio) 
z = translator.translate(s,'en') 
print(z.text)  
    
