
from embeddings import create_embeddings

documents = [
    "Machine learning allows computers to learn from data.",
    "Cloud computing provides computing resources over the internet.",
    "Neural networks are inspired by the human brain."
]

question = "How do computers learn from data?"

doc_vectors = create_embeddings(documents)
question_vector = create_embeddings([question])[0]

# Compare the question with each document
scores = doc_vectors @ question_vector

best_index = scores.argmax()

print("Question:", question)
print("Best matching document:", documents[best_index])
print("Similarity score:", round(float(scores[best_index]), 3))
