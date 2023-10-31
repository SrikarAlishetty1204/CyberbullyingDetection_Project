from flask import Flask, render_template, request,flash,redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload1():
    return render_template('select.html')

@app.route('/twitter')
def twitter():
    
    
    return render_template('upload.html')
@app.route('/wikipedia')
def wikipedia():
   
   
    return render_template('upload2.html')

@app.route('/upload', methods=['POST'])
def upload():
    global data
    global vectorizer
    global sv
    if request.method == 'POST':
        
        file = request.files['file']
        data = pd.read_csv(file)
        X = data['tweet']
        y = data['label']
        vectorizer = TfidfVectorizer(stop_words='english')
        X_vectorized = vectorizer.fit_transform(X)
        svm = SVC(kernel='poly')
        sv = svm.fit(X_vectorized, y)
        text='succesful'
        return render_template('upload.html',result=text)
@app.route('/upload2',methods=['POST'])
def upload2():
    global data
    global vectorizer
    global sv
    if request.method == 'POST':
        
        file = request.files['file']
        data = pd.read_csv(file)
        X = data['comment']
        y = data['label']
        vectorizer = TfidfVectorizer(stop_words='english')
        X_vectorized = vectorizer.fit_transform(X)
        svm = SVC(kernel='linear')
        sv = svm.fit(X_vectorized, y)
        text='succesful'
        return render_template('upload2.html',result=text)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/home2')
def home2():
    return render_template('home2.html')

@app.route('/predict', methods=['POST'])
def predict():
    global data
    tweet = request.form['a']
    user_vectorized = vectorizer.transform([tweet])
    predicted = sv.predict(user_vectorized)
    if predicted[0] == 1:
        result = "This tweet is cyberbullying."
    else:
        result = "This tweet is not cyberbullying."
    return render_template('home.html', prediction_text=result)

@app.route('/predict2',methods=['POST'])
def predict2():
    global data
    comment = request.form['a']
    user_vectorized = vectorizer.transform([comment])
    predicted = sv.predict(user_vectorized)
    if predicted[0] == 1:
        result = "This comment is cyberbullying."
    else:
        result = "This comment is not cyberbullying."
    return render_template('home2.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
