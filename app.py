import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import seaborn as sns
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from flask import Flask,render_template,request

app=Flask(__name__)

import pickle
model=pickle.load(open(r'C:/Users/trisha ghali/Flask/model.pkl','rb'))
tfidf_title = pickle.load(open(r'C:/Users/trisha ghali/Flask/tftitle.pkl','rb'))
tfidf_desc = pickle.load(open(r'C:/Users/trisha ghali/Flask/tfdesc.pkl','rb'))

def cleaner(data):
      data['Title'] = data['Title'].map(lambda x: x.lower())
      data['Description'] = data['Description'].map(lambda x: x.lower())
  
  # Remove numbers
      data['Title'] = data['Title'].map(lambda x: re.sub(r'\d+', '', x))
      data['Description'] = data['Description'].map(lambda x: re.sub(r'\d+', '', x))
  
  
  #Removing http
      data['Title'] = data['Title'].map(lambda x: re.sub(r'http\S+','',x))
      data['Description'] = data['Description'].map(lambda x: re.sub(r'http\S+','',x))
  
  #Removing https
      data['Title'] = data['Title'].map(lambda x: re.sub(r'https\S+','',x))
      data['Description'] = data['Description'].map(lambda x: re.sub(r'https\S+','',x))
  
  # Remove Punctuation
      data['Title']  = data['Title'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
      data['Description']  = data['Description'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
  
  # Remove white spaces
      data['Title'] = data['Title'].map(lambda x: x.strip())
      data['Description'] = data['Description'].map(lambda x: x.strip())
  
  # Tokenize into words
      data['Title'] = data['Title'].map(lambda x: word_tokenize(x))
      data['Description'] = data['Description'].map(lambda x: word_tokenize(x))
   
  # Remove non alphabetic tokens
      data['Title'] = data['Title'].map(lambda x: [word for word in x if word.isalpha()])
      data['Description'] = data['Description'].map(lambda x: [word for word in x if word.isalpha()])
  # filter out stop words
      stop_words = set(stopwords.words('english'))
      data['Title'] = data['Title'].map(lambda x: [w for w in x if not w in stop_words])
      data['Description'] = data['Description'].map(lambda x: [w for w in x if not w in stop_words])
  
  # Word Lemmatization
      lem = WordNetLemmatizer()
      data['Title'] = data['Title'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
      data['Description'] = data['Description'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
  
  # Turn lists back to string
      data['Title'] = data['Title'].map(lambda x: ' '.join(x))
      data['Description'] = data['Description'].map(lambda x: ' '.join(x))
  
      
  # Change to lowercase
      data['Title'] = data['Title'].map(lambda x: x.lower())
      data['Description'] = data['Description'].map(lambda x: x.lower())
  
  # Remove numbers
      data['Title'] = data['Title'].map(lambda x: re.sub(r'\d+', '', x))
      data['Description'] = data['Description'].map(lambda x: re.sub(r'\d+', '', x))
  
  
  #Removing http
      data['Title'] = data['Title'].map(lambda x: re.sub(r'http\S+','',x))
      data['Description'] = data['Description'].map(lambda x: re.sub(r'http\S+','',x))
  
  #Removing https
      data['Title'] = data['Title'].map(lambda x: re.sub(r'https\S+','',x))
      data['Description'] = data['Description'].map(lambda x: re.sub(r'https\S+','',x))
  
  # Remove Punctuation
      data['Title']  = data['Title'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
      data['Description']  = data['Description'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
  
  # Remove white spaces
      data['Title'] = data['Title'].map(lambda x: x.strip())
      data['Description'] = data['Description'].map(lambda x: x.strip())
  
  # Tokenize into words
      data['Title'] = data['Title'].map(lambda x: word_tokenize(x))
      data['Description'] = data['Description'].map(lambda x: word_tokenize(x))
   
  # Remove non alphabetic tokens
      data['Title'] = data['Title'].map(lambda x: [word for word in x if word.isalpha()])
      data['Description'] = data['Description'].map(lambda x: [word for word in x if word.isalpha()])
  # filter out stop words
      stop_words = set(stopwords.words('english'))
      data['Title'] = data['Title'].map(lambda x: [w for w in x if not w in stop_words])
      data['Description'] = data['Description'].map(lambda x: [w for w in x if not w in stop_words])
  
  # Word Lemmatization
      lem = WordNetLemmatizer()
      data['Title'] = data['Title'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
      data['Description'] = data['Description'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
  
  # Turn lists back to string
      data['Title'] = data['Title'].map(lambda x: ' '.join(x))
      data['Description'] = data['Description'].map(lambda x: ' '.join(x))
  
      return data
  
def prediction_res(data):
      
      title_features = tfidf_title.transform(data['Title']).toarray()
      desc_features = tfidf_desc.transform(data['Description']).toarray()
      res_features = np.concatenate([title_features,desc_features],axis = 1)
      return res_features
  


@app.route('/')
def prediction():
    return render_template('prediction.html')


@app.route('/login',methods=['POST'])
def login():
        p=request.form["t"]
        q=request.form["ds"]
        p=str(p)
        q=str(q)
        dic = {'Title':[p],'Description':[q]}
        dataa = pd.DataFrame(dic)
        res_features = prediction_res(cleaner(dataa))
        result_pred = model.predict(res_features)
        print(result_pred)
        def switch_statement(value):
            
            switch = {
            0: "Art and Music",
            1: "Food",
            2: "History",
            3: "Science and Technology",
            4: "Manufacturing",
            5: "Travel Blogs"
            }
            return switch.get(value, "Invalid option")
        r=switch_statement(result_pred[0])
        return render_template("prediction.html",y="the classification is "+ str(r))


if __name__=='__main__':
    app.run(debug=False)
    
    
   
    

    