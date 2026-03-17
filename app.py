from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

model = joblib.load("inspection_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            swl = float(request.form["swl"])
            location = request.form["location"].lower()
            description = request.form["description"].lower()
            manufacture = request.form["manufacture"].lower()

            data = pd.DataFrame({
                "SWL":[swl],
                "Location":[location],
                "Description":[description],
                "Manufacture":[manufacture]
            })

            prediction = model.predict(data)[0]

        except Exception as e:
            prediction = str(e)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
