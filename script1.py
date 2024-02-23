from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Preprocess function
def preprocess(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    # Remove stopwords
    stop_words = set(stopwords.words("arabic"))
    words = [word.lower() for word in words if word.lower() not in stop_words]
    return " ".join(words)

# Sample sentences
#sentence1 = "The quick brown fox jumps over the lazy dog."
sentence1 = "احمد يحب شرب اللبن."
#sentence2 = "A fast brown fox leaps over a sleepy canine."
sentence2 = "احمد يحب اللبن دائما"

# Preprocess the sentences
preprocessed_sentence1 = preprocess(sentence1)
preprocessed_sentence2 = preprocess(sentence2)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Calculate TF-IDF vectors for the sentences
tfidf_vectors = tfidf_vectorizer.fit_transform([preprocessed_sentence1, preprocessed_sentence2])

# Compute the cosine similarity
cosine_similarity_score = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

print("Cosine similarity score:", cosine_similarity_score)
