from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import csv

app = Flask(__name__)
CORS(app)

# Load model and intents
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

user_states = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get("sessionId")
    message = data.get("message")

    # Check if user is registered
    if session_id not in user_states or user_states[session_id].get("step") != "chat":
        return jsonify({"reply": "Please complete registration first."})

    sentence_tokens = tokenize(message)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return jsonify({"reply": response})
    else:
        return jsonify({"reply": "I'm not sure I understand. Can you please rephrase?"})

@app.route('/registerUser', methods=['POST'])
def register_user():
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")
    name = data.get("name")
    application = data.get("application")
    userid = data.get("userid")

    # Save user details to a CSV file
    with open('userdetails.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([email, phone, name, userid, application])

    # Mark user as registered in the session
    user_states[email] = {"step": "chat", "name": name, "application": application}

    return jsonify({"success": True})

@app.route('/checkUser', methods=['POST'])
def check_user():
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")

    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Email'] == email and row['Phone'] == phone:
                return jsonify({
                    "success": True,
                    "name": row['Name'],
                    "application": row['Application']
                })

    # User not found
    return jsonify({
        "success": False,
        "message": "User not found"
    })

if __name__ == '__main__':
    app.run(debug=True)
