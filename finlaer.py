import speech_recognition as sr 
import pyaudio
import wave

import nltk,string,numpy,math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import wave
import math
import nltk,string,numpy,math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

from nltk.stem.porter import PorterStemmer
    


'''from googletrans import Translator
translator = Translator()


z = translator.translate(s,'en') 
print(z.text)'''


#Similarity of texts

#
from googletrans import Translator
translator = Translator()
filename = (r"Testdrive.wav")
#def filer(filename)


AUDIO_FILE = (filename) #Iddar qwe.wav ke jagaha file ka naam dee


r = sr.Recognizer() 

with sr.AudioFile(AUDIO_FILE) as source: 
    audio = r.record(source) 
temp = r.recognize_google(audio)
try: 
    print("The audio file contains: " + temp) 


except sr.UnknownValueError: 
    print("Google Speech Recognition could not understand audio") 

except sr.RequestError as e: 
    print(e) 
print(temp,)
s=temp
h = translator.translate(s,'en') 
z = h.text
print(h.text)

porter_stemmer = PorterStemmer()






# Use TextBlob


# Use NLTK's PorterStemmer



lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    
def stemming_tokenizer(words):
    #words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words_1 = [porter_stemmer.stem(word) for word in words]
    return words_1    
    
def tot_tokenizer(text):
    return textblob_tokenizer(text,LemNormalize(text))

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


def cl(z):
    
    with open(r'C:\Users\freje\.spyder-py3\Symbiosis\enquiry.txt', 'r') as myfile1:
          enquiry_1 = myfile1.readlines()
    with open(r'C:\Users\freje\.spyder-py3\Symbiosis\breakdown.txt', 'r') as myfile2:
          breakdown_1 = myfile2.readlines()        
    with open(r'C:\Users\freje\.spyder-py3\Symbiosis\review.txt', 'r') as myfile3:
          feedback = myfile3.readlines()    
    with open(r'C:\Users\freje\.spyder-py3\Symbiosis\vehicle_quality.txt', 'r') as myfile1:
           vehicle_quality = myfile1.readlines()
    documents = []
    target = []
    for i in enquiry_1:
        documents.append(i)
        target.append('enquiry')
    for j in breakdown_1:
        documents.append(j)
        target.append('breakdown')
    for k in  feedback:
        documents.append(k)
        target.append('feedback')
    for i in vehicle_quality:
        documents.append(i)
        target.append('vehicle_qualiity')    

    '''
    
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
     
    LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english',analyzer = 'word')
    LemVectorizer.fit_transform(documents)
    #print("")
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
    #print("")
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    #print('Simaliraty using TD-IDF')
    #print (cos_similarity_matrix)
    #print(1)
    
    #print(documents)
    
    Tfidf_vect = TfidfVectorizer(max_features=5000, tokenizer=LemNormalize, stop_words='english',analyzer = 'word')
    Tfidf_vect.fit(documents)
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    new_doc = []
    new_doc.append(z)
    my_dict = {1:'enquiry',0:'breakdown',2:'feedback',3:'vehicle_quality'}
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
    tfidf_eg =  Tfidf_vect.transform(new_doc)
    predicted = SVM.predict(tfidf_eg)
    print(predicted)
    return predicted,my_dict
    '''
    #targets = ['enquiry', 'breakdown' , 'feedback','vehicle_quality']  
    Encoder = LabelEncoder()
    Y = Encoder.fit_transform(target)
    D = list(Y)
    Y = np.array(Y).reshape(-1, 1) 
    print(type(D))
    my_dict = {1:'enquiry',0:'breakdown',2:'feedback',3:'vehicle_quality'}
    print(Y)
    #Test_Y = Encoder.fit_transform(Test_Y)
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(documents,Y,test_size=0.3)
   # X = column_or_1d(X, warn=True)
   # Y = column_or_1d(Y, warn=True)
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemNormalize, stop_words='english',analyzer = 'word')),
                     ('tfidf', TfidfTransformer(norm="l2")),
                     ('clf', LinearSVC()),
                     ])
    text_clf.fit(Train_X,Train_Y)
    print(Train_Y)
    print('%')
    print(Test_Y)
    new_doc = []
    new_doc.append(z)
    #predicted = text_clf.predict()
    predicted = text_clf.predict(new_doc)
    pred_1 =  text_clf.predict(Test_X)
    #predicted = text_clf.predict(documents)
    print(pred_1,predicted)
    print(metrics.classification_report(Test_Y, pred_1))
    #print(confusion_matrix(Test_Y, pred_1))
    print("SVM Accuracy Score -> ",accuracy_score(pred_1,Test_Y)*100)
    return predicted,my_dict



def check_class(z,predicted_class,class_dict):
    label = class_dict[predicted_class]
    if z.lower().find('test drive') != -1:
        print('class = test_drive')
        j = 'test_drive'
        return j    
    elif z.lower().find('complain')  != -1:
        print('class = vehicle_quality ')
        f = 'vehicle_quality '
        return f 
    else :
        print('class:' + label )
        return label



cf,dicter = cl(z)
typer = check_class(z,cf[0],dicter)

import pandas as pd
filenames  = [] 
labels = []
my_dict = {'filename':filenames,'type': labels}
df = pd.DataFrame(my_dict , columns = ['filename','type'])
df.to_csv(r'I-K3QTT.csv')

def file_csv(filename,label):
#    z = filer(filename)
    cf,dicter = cl(filename)
    typer = check_class(filename,cf[0],dicter)
    df = df.append(filename,ignore_index=True)
    
file_csv(filename)            