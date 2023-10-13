import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')

# Sample data for text classification
documents = [
    ("This is a positive document.", "positive"),
    ("Negative sentiment is not good.", "negative"),
    ("I feel happy today.", "positive"),
    ("I don't like this product.", "negative"),
    ("Amazing performance!", "positive"),
    ("This is a terrible movie.", "negative"),
]

# Preprocessing: tokenization, stop words removal, and stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenized_documents = []

for (document, label) in documents:
    words = word_tokenize(document)
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    tokenized_documents.append((' '.join(words), label))

# Creating a bag of words (BoW) model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([doc for doc, _ in tokenized_documents])
y = [label for _, label in tokenized_documents]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate and print the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# You can use this classifier to classify new text
new_text = "I love this product, it's amazing!"
new_text = ' '.join([stemmer.stem(word) for word in word_tokenize(new_text) if word.lower() not in stop_words])
new_text_vectorized = vectorizer.transform([new_text])
predicted_label = classifier.predict(new_text_vectorized)
print(f"Predicted label for the new text: {predicted_label[0]}")
