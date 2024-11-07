from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import csv
import torch
import re
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and intents data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

ticket_counter = 1  # Initialize ticket counter
bot_name = "Helpdesk"

# Initialize ticket counter
def initialize_ticket_counter(filename):
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            tickets = [row[0] for row in reader if row]
            ticket_numbers = [int(re.search(r'\d+', ticket).group()) for ticket in tickets if re.search(r'\d+', ticket)]
            return max(ticket_numbers, default=0) + 1
    except FileNotFoundError:
        return 1

ticket_counter = initialize_ticket_counter('tickets.csv')

# Check if user is registered
def check_user_registration(email, phone):
    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Email'] == email and row['Phone'] == phone:
                return row
    return None

# Check if ticket already exists
def ticket_exists(user_id, application, problem_type):
    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (row['User ID'] == user_id and row['Application'] == application and row['Problem Type'] == problem_type):
                return True
    return False

# Route to handle chat messages from the frontend
@app.route('/chat', methods=['POST'])
def chat():
    global ticket_counter
    
    # Retrieve JSON data from frontend request
    data = request.json
    email = data.get('email')
    phone = data.get('phone')
    user_message = data.get('message')

    # Check user registration
    user_info = check_user_registration(email, phone)
    if not user_info:
        return jsonify({"reply": "You are not registered. Please register first."})

    # Process the message
    sentence_tokens = tokenize(user_message)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Generate bot response based on intent
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                return jsonify({"reply": response, "ticket_suggestion": "Would you like to raise a ticket?"})
    else:
        return jsonify({"reply": "I couldn't understand your issue. Would you like to raise a ticket for support?"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
