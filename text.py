#Python 2.x program to transcribe an Audio file 
import speech_recognition as sr 
import pyaudio
import wave

import nltk,string,numpy,math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


from googletrans import Translator
translator = Translator()

#
##Itta part hide karde Yeh recording ka part hai {
#FORMAT = pyaudio.paInt16
#CHANNELS = 2
#RATE = 44100
#CHUNK = 1024
#RECORD_SECONDS = 5
#WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
## start Recording
#stream = audio.open(format=FORMAT, channels=CHANNELS,
#                rate=RATE, input=True,
#                frames_per_buffer=CHUNK)
#print("recording...")
#frames = []
# 
#for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#    data = stream.read(CHUNK)
#    frames.append(data)
#print("finished recording")
# 
# 
## stop Recording
#stream.stop_stream()
#stream.close()
#audio.terminate()
# 
#waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#waveFile.setnchannels(CHANNELS)
#waveFile.setsampwidth(audio.get_sample_size(FORMAT))
#waveFile.setframerate(RATE)
#waveFile.writeframes(b''.join(frames))
#waveFile.close()
#
##} Yaha tak

AUDIO_FILE = (r"Testdrivemix.wav") #Iddar qwe.wav ke jagaha file ka naam dee

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
print(r.recognize_google(audio))
s=r.recognize_google(audio) 
z = translator.translate(s,'en') 
print(z.text)  
    
#Similarity of texts


lemmer = nltk.stem.WordNetLemmatizer()
       
def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    
def cosine(vect1,vect2):
    num=0
    den1=0
    den2=0
    for i in range(len(vect1)):
        num=num+vect1[i]*vect2[i]
        den1+=vect1[i]*vect1[i]
        den2+=vect2[i]*vect2[i]
    cos=num/(math.sqrt(den1)*math.sqrt(den2))    
    return(cos)
    

def cl():
    d1 = " Enquiry information on latest new or future or existing product features, car, cars, enquire, price, availability, closest showroom to drop in for purchase or exchange,"
    #keywords
    d3=', Calls for booking test drives, follow up calls with customers to schedule the same, confirmation that test drive has been done as per schedule or with delayed schedule '
    d2 = z.text
    documents = [d1,d3, d2]
    print(d1)
    print(d2)
    
    LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
    LemVectorizer.fit_transform(documents)
    print("")
    print (LemVectorizer.vocabulary_)
    temp=[]
    for val in LemVectorizer.vocabulary_:
        temp.append(val)
    print(pos_tag(temp))    
    print("")
    tf_matrix = LemVectorizer.transform(documents).toarray()
    tf_matrix.shape
    

    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    print("")
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    print('Simaliraty using TD-IDF')
    print (cos_similarity_matrix)
    print(1)

    return tf_matrix
def c2():
    cosine_matrix=cl()
    cosine_mat=[[cosine(i,j) for j in cosine_matrix] for i in cosine_matrix]
    print('Cosine Simaliraty')
    print(cosine_mat)
    
cf = c2()