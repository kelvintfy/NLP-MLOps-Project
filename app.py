import re
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from dotenv import dotenv_values
import database

labels  = ['anova', 'bayesian', 'classification', 'clustering', 'correlation',
        'distributions', 'hypothesis-testing', 'logistic',
        'machine-learning', 'mathematical-statistics', 'mixed-model',
        'multiple-regression', 'neural-networks', 'normal-distribution',
        'probability', 'r', 'regression', 'self-study',
        'statistical-significance', 'time-series']

config = dotenv_values(".env")

mlb = MultiLabelBinarizer()
mlb.fit_transform([(c,) for c in labels])

tokenizer = RegexpTokenizer(r"\w+")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')
nltk_stop_words = nltk.corpus.stopwords.words('english')
stopwords = list(nltk_stop_words) + \
    list(sklearn_stop_words) + list(spacy_stopwords)
stopwords = list(dict.fromkeys(stopwords))
lemmatizer = WordNetLemmatizer()
pipeline = joblib.load("svm.joblib")

DB = 'database/database.db'
database.createDB(DB)


def preprocessor(input_sentence):
    input_sentence = pd.DataFrame([input_sentence])[0]
    input_sentence = input_sentence.apply(
        lambda x: re.sub('<code>.*?</code>', '', x, flags=re.DOTALL))
    input_sentence = input_sentence.str.replace(r'<[^<]+?>', '', regex=True)
    input_sentence = input_sentence.astype(str)

    input_sentence = input_sentence.apply(tokenizer.tokenize)
    input_sentence = input_sentence.apply(
        lambda x: [w for w in x if all(ord(c) < 128 for c in w)])
    input_sentence = input_sentence.apply(
        lambda x: [word for word in x if not bool(re.search(r'\d', word))])
    input_sentence = input_sentence.apply(
        lambda x: [word for word in x if not bool(re.search(r'_', word))])

    input_sentence = input_sentence.map(
        lambda x: [word for word in x if word not in stopwords])
    input_sentence = input_sentence.map(lambda x: [word.lower() for word in x])

    input_sentence = input_sentence.apply(
        lambda x: [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in x])
    input_sentence = input_sentence.apply(
        lambda x: [lemmatizer.lemmatize(word, pos=wordnet.NOUN) for word in x])
    input_sentence = input_sentence.apply(
        lambda x: [lemmatizer.lemmatize(word, pos=wordnet.ADJ) for word in x])
    input_sentence = input_sentence.apply(
        lambda x: [lemmatizer.lemmatize(word, pos=wordnet.ADV) for word in x])
    input_sentence = input_sentence.apply(
        lambda x: (' '.join([str(word) for word in x])))

    return input_sentence.tolist()


def convert_to_tags(predicted_binaries):
    return mlb.inverse_transform(predicted_binaries)


def requestResults(kw):
    inputs = preprocessor(kw)
    y_pred = pipeline.predict(inputs)
    results = convert_to_tags(y_pred)

    database.sql_query(
        DB, "INSERT INTO records(input, output) VALUES (?, ?)", (kw, ",".join(results[0])))

    return results[0]


def output_distribution(data):
    output = []
    for row in data:
        classes = row[1].split(',')
        for c in classes:
            output.append(c)
    
    output = pd.DataFrame(output, columns=['class'])
    data = output.groupby('class').size().reset_index(name='count')

    label = str(data['class'].values.tolist())
    # label = label.replace(r"'", '')
    # print(label)
    return data['class'].values.tolist(), data['count'].tolist()


app = Flask(__name__)


@app.route('/')
def home():
    data = database.show_data(DB, "records")
    # Table Data
    table_data = pd.DataFrame(data, columns=['Input','Output','Timestamp'])
    table_data = [table_data.to_html(table_id='table', header="true", index=True)]
    # Charts
    bar_label, bar_data = output_distribution(data)
    values = {"table_data": table_data,
              "bar_label": bar_label,
              "bar_data": bar_data}
    return render_template('home.html', **values)


@app.route('/', methods=['POST'])
def get_data():
    if request.method == 'POST':
        kw = request.form['search']
        return redirect(url_for('success', kw=kw))


@app.route('/success/<kw>')
def success(kw):
    values = {"result": str(requestResults(kw))}
    return render_template('result.html', **values)
    # return "<xmp>" + str(requestResults(kw)) + " </xmp> "


@app.route('/clear')
def clearTable():
    database.sql_query(DB, "DELETE FROM records")
    return redirect('/')


@app.route('/hello')
def hello():
    return 'Hello World!\n'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=config["PORT"], debug=config["DEBUG"])
