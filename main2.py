import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
data =pd.read_csv('wikipedia.csv')
X=data['comment']
y=data['label']
vectorizer=TfidfVectorizer(stop_words='english')
X_vectorized=vectorizer.fit_transform(X)
svm=SVC(kernel='linear')
svm.fit(X_vectorized,y)

while(True):
    user_input=input("Enter a comment: ")
    user_vector=vectorizer.transform([user_input])
    predicted=svm.predict(user_vector)
    if predicted[0] == 1:
        print("the comment is bullying")
    else:
        print("the comment is not bullying")