import csv
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
uri = os.getenv('MONGO_URI')
client = MongoClient(uri, server_api=pymongo.server_api.ServerApi(
 version="1", strict=True, deprecation_errors=True))
db = client['user'] 
userdetails_collection = db['Users']
tickets_collection = db['Tickets']

def seed_userdetails():
    with open('userdetails.csv', mode='r') as file:
        reader = csv.DictReader(file)
        userdetails = []

        for row in reader:
            userdetails.append({
                "email": row["Email"],
                "phone": row["Phone"],
                "name": row["Name"],
                "userid": row["User_ID"],
                "application": row["Application"]
            })

        if userdetails:
            userdetails_collection.insert_many(userdetails)
            print(f"Inserted {len(userdetails)} user details into the database.")
        else:
            print("No user details found to insert.")

def seed_tickets():
    with open('tickets.csv', mode='r') as file:
        reader = csv.DictReader(file)
        tickets = []

        for row in reader:
            tickets.append({
                "ticket_number": row["Ticket Number"],
                "userid": row["User_ID"],
                "application": row["Application"],
                "problem_type": row["Problem_Type"],
                "description": row["Problem_Description"],
                "status": row["Status"]
            })

        if tickets:
            tickets_collection.insert_many(tickets)
            print(f"Inserted {len(tickets)} tickets into the database.")
        else:
            print("No tickets found to insert.")

def main():
    # Clear existing data (optional)
    userdetails_collection.delete_many({})
    tickets_collection.delete_many({})

    # Seed the collections
    seed_userdetails()
    seed_tickets()

if __name__ == "__main__":
    main()
