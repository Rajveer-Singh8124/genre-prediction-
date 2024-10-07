from flask import Flask, render_template, request
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

app = Flask(__name__)

def remove_whitespace(text):
    return text.str.strip()

def remove_punctuation(text):
    return text.str.replace('[^\w\s]', '', regex=True)

def word_token(text):
    return text.apply(lambda x: x.split())

def remove_stopwords(text):
    return text.apply(lambda x: [word for word in x if word not in stopwords.words("english")])

def stemming(text):
    return text.apply(lambda x: [ps.stem(word) for word in x ]) 


def sparse_to_csr(sparse_matrix):
    if isinstance(sparse_matrix, csr_matrix):
        return sparse_matrix
    return csr_matrix(sparse_matrix)

def string_fun(text):
    return text.astype(str).agg(''.join, axis=0)

pipe = pickle.load(open("pipe.pkl", "rb"))


nltk.download('stopwords')

ps = PorterStemmer()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
     
        text = request.form['text']
        
        
        df = pd.DataFrame([text], columns=['text'])
        
   
        predicted_genre = pipe.predict(df)[0]
        
        return render_template('index.html', prediction=predicted_genre, input_text=text)

if __name__ == '__main__':
    app.run(debug=True)
