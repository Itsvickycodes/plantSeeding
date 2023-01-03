from flask import Flask, render_template,request
import os
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('Model/plantseeding.h5')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    image = request.files['upload']
    # print(image)
    image.save(os.path.join("static", image.filename))

    user_img = cv2.imread('static/' + image.filename)
    # print(user_img.shape)
    user_img = cv2.resize(user_img, (150, 150))
    user_img = user_img/255.0
    user_img = user_img.reshape(1, 150, 150, 3)
    print(user_img.shape)

    x = model.predict(user_img)
    y = np.argmax(x, axis=1)
    print(y)
    if y == 0:
        y = "The leaf is diseased cotton leaf"
    elif y == 1:
        y = "The leaf is diseased cotton plant"
    elif y == 2:
        y = "The leaf is fresh cotton leaf"
    else:
        y = "The leaf is fresh cotton plant"

    print(y[1])
    return render_template("index.html", y=y, img_path='static/' + image.filename)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
"""
{% if y %}
        <img src="{{img_path}}">
        <p>We think it's a {{ y }}</p>
    {% endif %}
</body>

    print(y[1][1])
    return render_template("index.html", y=y, img_path='static/' + image.filename)
    """