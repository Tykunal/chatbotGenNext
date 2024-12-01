from dotenv import load_dotenv
import os
from pymongo import MongoClient
from preprocess import preprocess_text
from transformer_embeddings import transformer_embeddings
from similarity import calculate_similarity

load_dotenv()

def existing_ticket(input_sentence,stored_statements):
    # Preprocess input
    input_sentence_processed = " ".join(preprocess_text(input_sentence, use_stemming=False))
    stored_statements_processed = [" ".join(preprocess_text(s, use_stemming=False)) for s in stored_statements]

        # Generate embeddings
    input_embedding = transformer_embeddings([input_sentence_processed])[0]
    stored_embeddings = transformer_embeddings(stored_statements_processed)

    # Calculate similarity
    similarity_results = calculate_similarity(input_embedding, stored_embeddings, stored_statements)

    # Find the maximum cosine similarity
    max_similarity_result = max(similarity_results, key=lambda x: x['cosine_similarity'])
    max_similarity_percentage = max_similarity_result['cosine_similarity'] # Convert to percentage
    best_matching_statement = max_similarity_result['statement']

    # Output the result
    print(f"Best Matching Statement: {best_matching_statement}")
    print(f"Cosine Similarity: {max_similarity_percentage:.2f}%")
    sentenceMatched = best_matching_statement
    percent = max_similarity_percentage
    if percent>60:
        return True, sentenceMatched
    else:
        return False, ""


currentUser = "tykunal@12"
problem_type = "payment_issue"
inp = "I am having payment issues."
inp2 = "I am having login issues."

MONGO_URI = os.getenv("MONGO_URI") 
client = MongoClient(MONGO_URI)
db = client["user"] 
users_collection = db["Users"]
tickets_collection = db["Tickets"]

matchingDesc = tickets_collection.find({"userid": currentUser, "problem_type": problem_type})
descList = []
for d in matchingDesc:
    descList.append(d.get('description'))

print("This is the matching list: ",descList)

result,statement = existing_ticket(inp2,descList)

if result:
    print("You have ticket present with description: ", statement, ".")
    print("Would you still like to raise the ticket?")
else:
    print("Ticket Generated!")
