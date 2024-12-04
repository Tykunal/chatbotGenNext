import sys
import os

curPath = os.path.dirname(os.getcwd())
print("Printing address: ", curPath)

# Add the `scripts` directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from preprocess import preprocess_text
from transformer_embeddings import transformer_embeddings
from similarity import calculate_similarity

# # Read input
# with open("input_sentence.txt", "r") as f:
#     input_sentence = f.read().strip()

# with open("stored_sentence.txt", "r") as f:
#     stored_statements = [line.strip() for line in f]


input_sentence = 'fjjfj somethign with payment'
stored_statements = ['i am facing payments issue', 'i am having payments issues', 'facing error in payments','dealing errors in payments']

# Preprocess input
input_sentence_processed = " ".join(preprocess_text(input_sentence, use_stemming=False))
stored_statements_processed = [" ".join(preprocess_text(s, use_stemming=False)) for s in stored_statements]

# Generate embeddings
input_embedding = transformer_embeddings([input_sentence_processed])[0]
stored_embeddings = transformer_embeddings(stored_statements_processed)

# Calculate similarity
similarity_results = calculate_similarity(input_embedding, stored_embeddings, stored_statements)

# Separate cases based on similarity
high_similarity = []
medium_similarity = []

for result in similarity_results:
    cosine_similarity_percentage = result['cosine_similarity']  # Convert to percentage
    if cosine_similarity_percentage > 80:
        high_similarity.append(result)
    elif 50 <= cosine_similarity_percentage <= 80:
        medium_similarity.append(result)

# Output results
arrayMatched = []
if high_similarity:
    # Find the highest similarity statement
    max_high_similarity = max(high_similarity, key=lambda x: x['cosine_similarity'])
    max_high_statement = max_high_similarity['statement']
    max_high_value = max_high_similarity['cosine_similarity'] # Convert to percentage
    print(f"Highest Matching Statement: {max_high_statement}")
    print(f"Cosine Similarity: {max_high_value:.2f}%")
else:
    if medium_similarity:
        print("Statements with Similarity between 50% and 80%:")
        arrayMatched = []
        for result in medium_similarity:
            statement = result['statement']
            similarity_value = result['cosine_similarity']# Convert to percentage
            print(f"Statement: {statement}, Similarity: {similarity_value:.2f}%")
            arrayMatched.append(statement)


print(high_similarity)
print(medium_similarity)
print("Appended statements in arrayMatched:", arrayMatched)