import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    X = df['message']
    Y = df.iloc[:,4:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

    # data for bar plot
    category_pct = Y_train.mean().sort_values(ascending= False)
    category = category_pct.index.str.replace('_', ' ')

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category,
                    y=category_pct
                )
            ],

            'layout': {
                'title': {
                'text': 'Proportions of categories',
                'font': {'size': 18, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
            },
                'yaxis': {
                    'title': {
                        'text':"Percentage",
                        'font': {'size': 15, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                    },
                    'tickformat': ',.0%',
                    
                },
                'xaxis': {
                    'title': {
                        'text':"Category",
                        'font': {'size': 15, 'family': 'Arial', 'color': 'black', 'weight': 'bold'}
                    },
                    'tickangle': 45
                },
                'height': 800, 
                'width': 1100,
                'margin': {
                    'l': 150, 
                    'r': 100,
                    'b': 200, 
                    't': 100, 
                    'pad': 4 
                }
            }
        },


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()