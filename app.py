from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import csv

app = Flask(__name__)
# Allow CORS for all domains on all routes
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:3000"}})  

# before situation
# CORS(app)
currentUser = ""
adminList = ["tykunal@12"]

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

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message")
    registered = data.get("isUserRegistered")
    # email = data.get("email")  # Get the user's email from the request data

    if not registered:
        return jsonify({"reply": "Please complete Login/Registration First."})

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
                return jsonify({
                    "reply": response,
                    "question": "Do you want to raise a ticket?",
                    "tag": tag
                })
    else:
        return jsonify({
            "reply": "It seems you are in some new kind of problem, wanna raise a ticket to be handled by backend?",
            "tag": "New"
        })

def get_last_ticket_number():
    last_ticket_number = 0 

    try:
        with open('tickets.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                last_ticket_number = row['Ticket Number']  # Get the ticket number from the last row
            
        if last_ticket_number:
            last_ticket_number = int(last_ticket_number.split('-')[-1])
        else:
            last_ticket_number = 0 

    except FileNotFoundError:
        last_ticket_number = 0
        
    return last_ticket_number + 1


@app.route('/raiseTicket', methods=['POST', 'OPTIONS'])
@cross_origin()
# @cross_origin(origin='http://127.0.0.1:3000')
def raise_ticket():
    if request.method == 'OPTIONS':
        return '', 200  
    application = ""
    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['User_ID'] == currentUser:
                application = row["Application"]

    data = request.get_json()
    problem_type = data.get("tag")
    description = data.get("description")

    ticket,status = ticket_exists(problem_type)
    if ticket:
        return jsonify({"reply": f"Your {ticket} already exists with Status {status}, thanks for visiting."})

    newTicketNumber = get_last_ticket_number();
    ticket_number = f"TICKET-{newTicketNumber}"
    # ticket_counter += 1

    # Store the ticket information in a CSV file
    with open('tickets.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([ticket_number, currentUser, application, problem_type, description, "Unresolved"])

    return jsonify({
        "reply": f"Your ticket is generated and ticket number is {ticket_number}, for application {application}. Thanks for visiting"
    })

def ticket_exists(problem_type):
    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['User_ID'] == currentUser and row['Problem_Type'] == problem_type:
                return row['Ticket Number'], row['Status']
    return None


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
    global currentUser #unless it will not update the value, since it will be limited to the checkUser function only.
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")

    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Email'] == email and row['Phone'] == phone:
                currentUser = row["User_ID"]
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

@app.route('/uniqueId', methods=['POST'])
def unique_id():
    data = request.get_json()
    id = data.get("userid")

    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['User_ID'] == id:
                return jsonify({
                    "success": False,
                    "message": "Please enter a different userid"
                })

    # User not found
    return jsonify({
        "success": True,
        "message": "UserId accepted!"
    })

@app.route('/admin', methods=['GET'])
@cross_origin()
def admin():
    if currentUser in adminList: 
        tickets = [] 
        with open('tickets.csv', mode='r') as file:
            reader = csv.DictReader(file)  
            for row in reader:
                tickets.append(row) 

        # return render_template("admin.html", tickets=tickets)
        return jsonify({"tickets": tickets})  
    else:
        return jsonify({"error": "Unauthorized access"}), 403 

if __name__ == '__main__':
    app.run(debug=True)
