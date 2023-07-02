from flask import Flask

app = Flask(__name__)


@app.route('/news', methods=['GET'])
def get_news():
    return "news api called"


def main():
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
