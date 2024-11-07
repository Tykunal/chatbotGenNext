from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import csv
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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

# Helper functions for user registration and ticket checking
def check_user_registration(email, phone):
    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Email'] == email and row['Phone'] == phone:
                return row
    return None

def register_user(details):
    with open('userdetails.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(details)

# State to manage user registration
user_states = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get("sessionId")  # Note the change to match front end
    message = data.get("message")

    # Initialize state if new session
    if session_id not in user_states:
        user_states[session_id] = {"step": "email"}

    state = user_states[session_id]

    # Step-by-step registration process
    if state["step"] == "email":
        state["email"] = message
        state["step"] = "phone"
        return jsonify({"reply": "Please enter your phone number."})

    elif state["step"] == "phone":
        state["phone"] = message
        user_info = check_user_registration(state["email"], state["phone"])
        if user_info:
            state["step"] = "chat"
            state["user_info"] = user_info
            return jsonify({"reply": f"Welcome back {user_info['Name']}! How can I assist you today?"})
        else:
            state["step"] = "register_name"
            return jsonify({"reply": "You are not registered. Please enter your name to register."})

    elif state["step"] == "register_name":
        state["name"] = message
        state["step"] = "register_user_id"
        return jsonify({"reply": "Please enter your User ID."})

    elif state["step"] == "register_user_id":
        state["user_id"] = message
        state["step"] = "register_application"
        return jsonify({"reply": "Please enter your application (Pensire, Vastuteq, or Procu)."})

    elif state["step"] == "register_application":
        state["application"] = message
        register_user([state["email"], state["phone"], state["name"], state["user_id"], state["application"]])
        state["step"] = "chat"
        return jsonify({"reply": f"Thank you {state['name']}, you are now registered! How can I help you?"})

    # Chat function logic after registration
    elif state["step"] == "chat":
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

if __name__ == '__main__':
    app.run(debug=True)
