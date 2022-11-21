#from flask import Flask, jsonify
#
#app = Flask('FinTechExplained WebServer')
#
#@app.route('/')
#def get_data():
#    return jsonify(1)
#
#if __name__ == '__main__':
#    app.run()

from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024