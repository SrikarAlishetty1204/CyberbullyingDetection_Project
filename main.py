import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
data = pd.read_csv('tweets_preprocessed.csv')
X = data['tweet']
y = data['label']
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
svm = SVC(kernel='poly')
sv=svm.fit(X_vectorized, y)
pickle.dump(sv,open('model.pkl','wb'))
pickle.dump(vectorizer, open('vectorizer.pickle', 'wb')) 

while(True):
 
    user_input = input("Enter a tweet: ")

   
    user_vectorized = vectorizer.transform([user_input])

    
    predicted = svm.predict(user_vectorized)

 
    if predicted[0] == 1:
        print("The tweet is cyberbullying.")
    else:
        print("The tweet is not cyberbullying.")
   