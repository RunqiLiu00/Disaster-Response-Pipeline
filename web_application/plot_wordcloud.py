import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(text):

    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in stopwords.words("english") and w.isalpha()]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
X = df['message']
Y = df.iloc[:,4:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

text = " ".join(X_train.tolist())
tokens = tokenize(text)
new_text = " ".join(tokens)

wc = WordCloud(width=1200, 
               height=800, 
               background_color='white',
               max_words=150
              )
wc.generate(new_text)
wc.to_file('app/static/wordcloud_train.png')