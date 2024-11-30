from scipy.spatial.distance import cosine, euclidean
import numpy as np

def calculate_similarity(input_embedding, stored_embeddings, stored_statements):
    similarity_scores = []
    for i, statement_embedding in enumerate(stored_embeddings):
        cosine_sim = 1 - cosine(input_embedding, statement_embedding)
        euclidean_dist = euclidean(input_embedding, statement_embedding)
        dot_product = np.dot(input_embedding, statement_embedding)
        similarity_scores.append({
            "statement": stored_statements[i],
            "cosine_similarity": round(cosine_sim * 100, 2),  # Percentage
            "euclidean_distance": round(euclidean_dist, 2),
            "dot_product": round(dot_product, 2)
        })
    return similarity_scores