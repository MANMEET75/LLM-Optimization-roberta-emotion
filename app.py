from flask import Flask, request, jsonify, render_template
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import time
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from src.exception import *
import os

app = Flask(__name__)

# Load the base RoBERTa model and tokenizer
base_model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

# Load the quantized model (you should have it saved and loaded here)
quantized_model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.float16)

# Preprocess text (username and link placeholders)
def preprocess(text):
    try:
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    except Exception as e:
        raise CustomException(e, sys)



@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        raise CustomException(e, sys)
    

@app.route('/predict_base', methods=['POST'])
def predict_base():
    try:
        text = request.form.get('text')

        task = 'emotion'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL)

        base_model_size = sum(p.numel() for p in base_model.parameters())

        base_model_size_mb = base_model_size * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter (32-bit float)

        
        # text = data['text']
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')

        # Measure latency for inference
        start_time = time.time()
        output = model(**encoded_input)
        latency = time.time() - start_time

        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        # Pass the relevant data to the HTML template
        return render_template('predict_base.html',base_model_size=base_model_size_mb,labels=labels, scores=scores, text=text, latency=latency)

    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict_optimized', methods=['POST'])
def predict_optimized():
    try:
        text = request.form.get('text')

        task = 'emotion'
        MODEL = "cardiffnlp/twitter-roberta-base-emotion"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        # text = data['text']
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        quantized_model_size = sum(p.numel() for p in quantized_model.parameters())
        quantized_model_size_mb = quantized_model_size * 2 / (1024 ** 2)  # Assuming 2 bytes per parameter (16-bit float)



        # Measure latency for inference
        start_time = time.time()
        output = quantized_model(**encoded_input)
        latency = time.time() - start_time
        
        output = quantized_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        # Pass the relevant data to the HTML template
        return render_template('predict_optimized.html',quantized_model_size=quantized_model_size_mb,labels=labels, scores=scores, text=text, latency=latency)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
