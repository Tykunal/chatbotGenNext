from pymongo import MongoClient
try:
    # start example code here
    uri = "mongodb+srv://kunaltyagi00000:r0MxNeESPhuhlWsy@cluster0.dbdkr.mongodb.net/user?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri)
    db = client['user'] 
    userdetails_collection = db['Users']
    tickets_collection = db['Tickets']

    # end example code here

    client.admin.command("ping")
    print("Connected successfully")

    # other application code

    client.close()

except Exception as e:
    raise Exception(
        "The following error occurred: ", e)
