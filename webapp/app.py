from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
import pandas as pd

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder="templates")

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    #data = model.named_steps["preprocessor"].transform(data)
    prediction = model.predict(data)
    return prediction

def api_response(request):
    try:
        data = np.array([list(request.json.values())])
        response = predict(data)
        response = {"response": response}
        return response
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong try again"}
        return render_template("404.html", error=error)



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                data = dict(request.form)
                data = pd.DataFrame(data, index=[1])
                print(data)
                response = predict(data)[0]
                return render_template("index.html", response=response)

            elif request.json:
                response = api_response(request)
                return jsonify(response)

        except Exception as e:
            print(e)
            error = {"error": "Something went wrong try again"}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)