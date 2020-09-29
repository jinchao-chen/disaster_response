import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # Grahp 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # Graph 2
    label_names = df.columns[4::]
    cat_columns = [cat_name  for cat_name in label_names if '_related' in cat_name]
    cat_counts = df[cat_columns].sum()
    cat_names = list(cat_counts.index)

    # Graph 3
    lable_counts = df[label_names].sum()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

            # Graph 1 - 
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Graph 2 - 
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Related Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related category"
                }
            }
        },
        # Graph 3-
        {
            'data': [
                Bar(
                    x=label_names,
                    y=lable_counts.values
                )
            ],

            'layout': {
                'title': 'Counts of Each Labels',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Labels"
                }
            }
        }
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
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()