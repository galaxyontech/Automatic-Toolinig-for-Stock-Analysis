from flask import Flask

app = Flask(__name__)


@app.route('/news', methods=['GET'])
def get_news():
    # your code to fetch and return news here
    pass


if __name__ == '__main__':
    app.run(debug=True)
