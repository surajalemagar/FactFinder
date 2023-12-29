import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data=pd.read_csv('news.csv')
print(data.head())
data.isnull().sum()  #checking for null value
data.drop(['Unnamed: 0','title'],axis=1,inplace=True)
data['label']=data['label'].map({'REAL':0,'FAKE':1})
print(data.head())

def preprocessing(text):
    text = text.strip()                         
    text = text.lower()                         
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)   
    
    return text

data['Clean']=data['text'].apply(preprocessing)

X_train, x_test, y_train, y_test = train_test_split(data['Clean'], 
                                                    data['label'],test_size=0.2, 
                                                    random_state=42)


vectorizer = CountVectorizer(binary=True, max_features=800)  
X_train_binary = vectorizer.fit_transform(X_train)
x_test_binary = vectorizer.transform(x_test)

df_binary = pd.DataFrame(X_train_binary.toarray(), columns=vectorizer.get_feature_names_out())


vectorizer = CountVectorizer(max_features=500)  

X_train_count = vectorizer.fit_transform(X_train)
x_test_count = vectorizer.transform(x_test)
df_train_count = pd.DataFrame(X_train_count.toarray(), columns=vectorizer.get_feature_names_out())

df_train_count.head()

vectorizer = TfidfVectorizer(max_features=10000)  

X_train_tfidf = vectorizer.fit_transform(X_train)
x_test_tfidf= vectorizer.transform(x_test)
df_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

X = X_train_tfidf
y = y_train
clf = LogisticRegression()
clf.fit(X, y)
y_pred=clf.predict(x_test_tfidf)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(x_test_tfidf)
clf.fit(X_train_scaled, y_train)
y_pred=clf.predict(X_test_scaled)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test,y_pred))