from preprocess import preprocess_text
from transformer_embeddings import transformer_embeddings
from similarity import calculate_similarity

# Read input
with open("input_sentence.txt", "r") as f:
    input_sentence = f.read().strip()

with open("stored_sentence.txt", "r") as f:
    stored_statements = [line.strip() for line in f]

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

# Write max result to file
with open('max_similarity_result.txt', "w") as f:
    f.write(f"Best Matching Statement: {best_matching_statement}\n")
    f.write(f"Cosine Similarity: {max_similarity_percentage:.2f}%\n")
