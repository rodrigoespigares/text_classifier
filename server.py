from flask import Flask, request, jsonify
from train import train_model
from classify import classify_text
import os

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    user_id = data['user_id']
    categories = data['categories']  # { "trabajo": ["tengo que trabajar", "mi empleo..."], ... }

    texts = []
    labels = []
    category_names = list(categories.keys())
    for idx, (cat, examples) in enumerate(categories.items()):
        texts.extend(examples)
        labels.extend([idx] * len(examples))

    train_model(user_id, texts, labels, category_names)
    return jsonify({"status": "trained", "categories": category_names})

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    user_id = data['user_id']
    text = data['text']
    category = classify_text(user_id, text)
    return jsonify({"category": category})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)