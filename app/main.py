from flask import Flask, render_template, request
import pandas as pd
from transformers import pipeline,logging
logging.set_verbosity_error()

model_path = "nlp_model/"
summarizer = pipeline("summarization", model=model_path, device="cpu",max_length=128)

app = Flask(__name__)

@app.route("/")
def hello_world(data=None,model_response=None):
    # return "<p>Hello, World!</p>"
    return render_template("page.html",data=data,model_response=model_response)


# @app.route('/get_rap_data', methods =["GET", "POST"])
@app.post('/get_rap_data')
def get_rap_data():
    # data=[{"a":"1","b":"Abid"},{"a":"2","b":"Abid2"},{"a":"3","b":"Abid3"}]
    df = pd.DataFrame([{"a":"1","b":"Abid"},{"a":"2","b":"Abid2"},{"a":"3","b":"Abid3"}])
    data = df.to_dict(orient="records")
    prompt = request.form["prompt"]
    model_response = summarizer(prompt)[0]["summary_text"]
    print("prompt",prompt,"response",model_response)
    return hello_world(data,model_response)

