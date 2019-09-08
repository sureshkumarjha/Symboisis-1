


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
    


'''from googletrans import Translator
translator = Translator()


z = translator.translate(s,'en') 
print(z.text)'''

z = input()

#Similarity of texts




from nltk.stem.porter import PorterStemmer
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
    enquiry = " Enquiry / see / view /  information on latest new or future or existing product features  price  availability of vehicle or car closest showroom to drop in for purchase or exchange  showroom price "
    #keywords
    test_drives ='Calls for booking test drives, follow up calls with customers to schedule the same, confirmation that test drive has been done as per schedule or with delayed schedule car model test drives'
    breakdown = 'customer calling for car breakdown , providing location details requesting assistance mechanic reaching till customer prelimnary input on vehicle condition engine oil loss glass break tyre puncture gear break accident smoke coming from vehicle or car or bike or motor  dents fire break failure nuts and bolts , help '
    #feedback = 'feedback collected post sales on vehicle car bike motor delivery on customer sales service , complains about service , praising services , good customer service , help  '
   
        
    test_drives ='Calls for booking test drives, follow up calls with customers to schedule the same, confirmation that test drive has been done as per schedule or with delayed schedule car model test drives'
    
    with open(r'C:\Users\user\Desktop\enquiry.txt', 'r') as myfile1:
          enquiry_1 = myfile1.readlines()
    with open(r'C:\Users\user\Desktop\breakdown.txt', 'r') as myfile2:
          breakdown_1 = myfile2.readlines()        
    with open(r'C:\Users\user\Desktop\review.txt', 'r') as myfile3:
          feedback = myfile3.readlines()       
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
    

    ''''LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english',analyzer = 'word')
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
    print("")
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    print('Simaliraty using TD-IDF')
    print (cos_similarity_matrix)
    #print(1)

    return tf_matrix'''
    print(documents)
    #target = ['enquiry', 'breakdown' , 'feedback']  
    Encoder = LabelEncoder()
    Y = Encoder.fit_transform(target)
    Y = np.array(Y).reshape(-1, 1)
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
    print(predicted)
    print(metrics.classification_report(Test_Y, pred_1))
    print(confusion_matrix(Test_Y, pred_1))
    print("SVM Accuracy Score -> ",accuracy_score(pred_1,Test_Y)*100)
    ''' 
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(documents,Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y)*100)
    '''
def c2(z):
    cosine_matrix=cl(z)
    cosine_mat=[[cosine(i,j) for j in cosine_matrix] for i in cosine_matrix]
    #print('Cosine Simaliraty')
    return cosine_mat
    
def determine_class(z,threshold):
    cf = c2(z)
    #print(cf)
    x = cf[0]
    r = max(x[1:])
    #print(r,1)
    index = 0
    a = z.find('test drive')
    print(a)
    for i in x:
        if i == r:
            break
        elif r < threshold:
            break;
        else:
            index = index + 1 
            
    print(index)     
    if index == 1 :
        print(z + '-' + 'class:enquiry')
    elif index == 2 and a != -1:
        print(z + '-' + 'class:test drive')
    elif index == 3:
        print(z + '-'+ 'class:breakdown')
    elif index == 4:
        print(z + '-' + 'class:feedback')
    else:
        print('no class assigned')

