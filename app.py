from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import base64
# from utils import *


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app)


@app.route("/", methods=["GET", "POST"])
def main():
    return send_file("./static/html/index.html")


@app.route("/process", methods=["POST"])
def process_images():

    if request.method == "POST":

        return jsonify(msg="done")


if __name__ == "__main__":
    app.run(debug=True)
