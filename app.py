from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS, cross_origin
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import csv
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Allow CORS for all domains on all routes
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:3000"}})  

# before situation
# CORS(app)

# currentUser = ""
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)
adminList = ["tykunal@12", "tykunal@12345"]
app.secret_key = os.getenv("SECRET_KEY") 

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


# session route
@app.before_request
def make_session_permanent():
    session.permanent = True


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_currentUser', methods=['GET'])
def get_current_user():
    currentUser = session.get('userid')
    if currentUser in adminList:
        current_user = currentUser
        return jsonify({'currentUser': current_user})
    return jsonify({'error': 'No current user or unauthorized access'}), 403

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


@app.route('/raiseTicket', methods=['POST'])
# @cross_origin()
def raise_ticket():
    # if request.method == 'OPTIONS':  #added OPTIONS inside methods.
    #     return '', 200  
    application = ""
    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['User_ID'] == session.get('userid'):
                application = row["Application"]

    data = request.get_json()
    problem_type = data.get("tag")
    description = data.get("description")

    result = ticket_exists(problem_type)
    if result is not None:
        ticket, status = result
        return jsonify({"reply": f"Your {ticket} already exists with Status {status}, thanks for visiting."})

    newTicketNumber = get_last_ticket_number();
    ticket_number = f"TICKET-{newTicketNumber}"

    # Store the ticket information in a CSV file
    with open('tickets.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([ticket_number, session['userid'], application, problem_type, description, "Unresolved"])

    return jsonify({
        "reply": f"Your ticket is generated and ticket number is {ticket_number}, for application {application}. Thanks for visiting"
    })

def ticket_exists(problem_type):
    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        currentUser = session.get('userid')
        for row in reader:
            if row['User_ID'] == currentUser and row['Problem_Type'] == problem_type:
                return row['Ticket Number'], row['Status']
    return None


@app.route('/registerUser', methods=['POST'])
def register_user():
    # global currentUser
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")
    name = data.get("name")
    application = data.get("application")
    userid = data.get("userid")
    
    with open('userdetails.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([email, phone, name, userid, application])

    session.permanent = True
    session['userid']= userid
    return jsonify({"success": True})

@app.route('/checkUser', methods=['POST'])
def check_user():
    # global currentUser #unless it will not update the value, since it will be limited to the checkUser function only.
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")

    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Email'] == email and row['Phone'] == phone:
                session['userid'] = row["User_ID"]
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

@app.route('/admin')
def admin():
    currentUser = session.get('userid')
    if currentUser in adminList:
        return render_template('admin.html')
    return "Access denied. You are not authorized to view this page.", 403



@app.route('/adminData', methods=['GET'])
@cross_origin()
def adminData():
    currentUser = session.get('userid')
    # currentUser = 'tykunal@12345'
    if currentUser not in adminList:
        return jsonify({'error': 'Unauthorized access'}), 403

    tickets = []
    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tickets.append(row)

    return jsonify(tickets)

# deletion and updation logic

@app.route('/deleteTicket', methods=['DELETE'])
def delete_ticket():
    ticket_number = request.args.get('ticketNumber')
    try:
        tickets = []
        with open("tickets.csv", "r") as file:
            reader = csv.DictReader(file)
            tickets = list(reader)

        # Check if the ticket exists and remove it
        ticket_found = False
        updated_tickets = []
        for ticket in tickets:
            if ticket["Ticket Number"] == ticket_number:
                ticket_found = True
            else:
                updated_tickets.append(ticket)

        if not ticket_found:
            return jsonify({"error": f"Ticket {ticket_number} not found."}), 404

        # Write the updated tickets back to the CSV file
        with open("tickets.csv", "w", newline="") as file:
            fieldnames = ["Ticket Number", "User_ID", "Application", "Problem_Type", "Problem_Description", "Status"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_tickets)

        return jsonify({"message": f"Ticket {ticket_number} deleted successfully."}), 200

    except FileNotFoundError:
        return jsonify({"error": "Tickets file not found."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/edit')
def edit():
    if session.get('userid') in adminList:
        return render_template('edit.html')
    else:
        return "Access denied. You are not authorized to view this page.", 403

@app.route('/getEditFormData', methods=['POST'])
def getEditFormData():
    data = request.get_json()
    ticketNumber = data.get("ticketNumber")
    if not ticketNumber:
        return jsonify({"error": "Ticket number is required"}), 400

    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Ticket Number'] == ticketNumber:
                row.pop('Ticket Number', None)
                return jsonify(row), 200

    return jsonify({"error": "Ticket not found"}), 404
      
    

@app.route('/editTicket', methods=['POST'])
def editTicket():
    # Parse JSON data from the request
    data = request.get_json()
    ticketNumber = data.get("ticketNumber")
    # description = data.get("problemDescription")
    # problemType = data.get("problemType")
    # status = data.get("status")

    if not ticketNumber:
        return jsonify({"error": "Ticket number is required"}), 400

    updated = False
    # Read the existing data
    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    with open('tickets.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row['Ticket Number'] == ticketNumber:
                # Update only fields provided in the request
                if "problemDescription" in data:
                    row['Problem_Description'] = data['problemDescription']
                if "problemType" in data:
                    row['Problem_Type'] = data['problemType']
                if "status" in data:
                    row['Status'] = data['status']
                updated = True
            writer.writerow(row)

    if updated:
        return jsonify({"message": f"{ticketNumber} updated successfully"}), 200
    else:
        return jsonify({"error": "Ticket not found"}), 404

# Ensuring that the user doesnot have to relogin till the session exits.

@app.route('/isLogged', methods=['GET'])
def is_logged():
    if session.get('userid'):
        return jsonify({"isLoggedIn": True}), 200 
    else:
        return jsonify({"isLoggedIn": False}), 200


if __name__ == '__main__':
    app.run(debug=True)
