from flask import Flask, request, jsonify
from Code.NewSearch.news_search import NewSearch

app = Flask(__name__)


@app.route('/')
def home():
    return "OK"


@app.route('/get-news', methods=['GET'])
def get_news():
    # your code to fetch and return news here
    stocks = request.args.getlist('stocks')
    print(stocks)
    news_searcher = NewSearch(stocks)
    news_searcher.search_key_words()
    return jsonify(stocks)


@app.route('/get-news/get-summary', methods=['GET']) 
def get_news():
    # your code to fetch and return news here
    stocks = request.args.getlist('stocks')
    print(stocks)
    news_searcher = NewSearch(stocks)
    news_searcher.search_key_words()
    return jsonify(stocks)

# @app.route('')
def main():
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
