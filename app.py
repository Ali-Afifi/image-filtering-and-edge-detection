from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import base64
from utils import *


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app)


@app.route("/", methods=["GET", "POST"])
def main():
    return send_file("./static/html/index.html")


@app.route('/download')
def downloadFile():
    return send_file(f"./output/output.jpeg", mimetype="image/png")


@app.route("/process", methods=["POST"])
def process_images():

    if request.method == "POST":
        data = request.get_json()

        with open("./input/input.jpeg", 'wb') as f:
            f.write(base64.b64decode(data["image1"].split(",")[1]))

        option1 = data["option1"]

        if option1 == 1:
            add_gaussian_noise()
        elif option1 == 2:
            filter_gaussian_noise()
        elif option1 == 3:
            sobel()
        elif option1 == 4:

            option2 = data["option2"]

            if option2 == 1:
                draw_histogram()
            elif option2 == 2:
                draw_distribution_curve()

        elif option1 == 5:
            draw_histogram_equalization()
        elif option1 == 6:
            normalize()
        elif option1 == 7:

            option2 = data["option2"]

            if option2 == 1:
                global_threshold()
            elif option2 == 2:
                Local_threshold()

        elif option1 == 8:
            gray_scale()

        binary_fc = open("./output/output.jpeg", 'rb').read()
        base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
        dataurl = f'data:image/png;base64,{base64_utf8_str}'

        return jsonify(msg="done", img=dataurl)


if __name__ == "__main__":
    app.run(debug=True)
