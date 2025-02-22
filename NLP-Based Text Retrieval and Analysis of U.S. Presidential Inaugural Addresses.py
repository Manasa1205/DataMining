
# -*- coding: utf-8 -*-


import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict
import math

# Directory for the corpus files
corpus_dir = '/Users/manasavardhini/Desktop/P1/US_Inaugural_Addresses'

# Variables to store document and term statistics
doc_freq = Counter()  # Stores the document frequency of each term
term_freq = {}        # Stores term frequency for each document
doc_vector = {}       # Stores the normalized TF-IDF vector for each document
total_docs = 0        # Keeps track of the number of documents

# Tokenizer for processing words
word_tokenizer = RegexpTokenizer(r'[a-zA-Z]+')  # Tokenizes the document into alphabetic words

# Loop through all text files in the corpus directory
for filename in os.listdir(corpus_dir):
    if filename.startswith(('0', '1', '2', '3')):  # Only process files starting with '0', '1', '2', '3'
        with open(os.path.join(corpus_dir, filename), "r", encoding='windows-1252') as file:
            content = file.read().lower()  # Read and convert content to lowercase

        # Tokenize the text content and remove stopwords
        tokens = word_tokenizer.tokenize(content)  # Extract words from text
        stop_words = set(stopwords.words('english'))  # Load English stopwords
        filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords

        # Apply stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]  # Apply stemming to tokens

        total_docs += 1  # Increment document count

        # Update document frequency and term frequency
        doc_freq.update(set(stemmed_tokens))  # Count how many documents contain each term (document frequency)
        term_freq[filename] = Counter(stemmed_tokens)  # Count term frequency for each document

# Function to get the normalized TF-IDF weights for a term in a document
def getweights(filename, term):
    return doc_vector[filename].get(term, 0)  # Return the TF-IDF weight if it exists, otherwise return 0

# Function to get the weight of a term after stemming
def getweight(filename, term):
    stemmed_term = stemmer.stem(term)  # Stem the term
    return doc_vector[filename].get(stemmed_term, 0)  # Return the TF-IDF weight for the stemmed term

# Function to compute the inverse document frequency (IDF) of a term
def getidf(term):
    stemmed_term = stemmer.stem(term)  # Stem the term
    if doc_freq[stemmed_term]:
        # Calculate IDF as log10 of total documents divided by the number of documents containing the term
        return math.log10(len(term_freq) / doc_freq[stemmed_term])
    return -1  # Return -1 if the term doesn't appear in any documents

# Function to calculate TF-IDF weights for a term in a document
def compute_tf_idf(filename, stemmed_term):
    if term_freq[filename][stemmed_term]:
        # Calculate the term frequency (TF) and multiply by IDF to get TF-IDF
        return (1 + math.log10(term_freq[filename][stemmed_term])) * getidf(stemmed_term)
    return 0  # Return 0 if the term doesn't appear in the document

# Normalize the TF-IDF weight vector for a document
def normalize_vector(weight_vector):
    normalized_vector = {}
    magnitude = math.sqrt(sum(weight ** 2 for weight in weight_vector.values()))  # Calculate vector magnitude
    for term, weight in weight_vector.items():
        normalized_vector[term] = weight / magnitude  # Normalize each weight by dividing by the magnitude
    return normalized_vector  # Return the normalized vector

# Compute and normalize the TF-IDF vectors for each document
for filename in term_freq:
    weight_vector = {token: compute_tf_idf(filename, token) for token in term_freq[filename]}  # Compute TF-IDF for each term
    doc_vector[filename] = normalize_vector(weight_vector)  # Normalize the TF-IDF vector and store it

# Build the postings list for each term, sorted by weight
postings = defaultdict(list)  # Postings list: maps terms to a list of (document, weight) pairs
for filename in term_freq:
    for term in term_freq[filename]:
        weight = doc_vector[filename][term]
        postings[term].append((filename, weight))  # Add (document, weight) for each term

for term in postings:
    postings[term].sort(key=lambda x: x[1], reverse=True)  # Sort each term's postings list by weight (descending)

# Query function using postings list to get varied results
def query_with_postings(query_str):
    query_tokens = query_str.lower().split()  # Convert the query string to lowercase and split into words
    query_weights = {}  # Stores the query's TF-IDF weights
    query_length = 0  # Tracks the query vector's length (for normalization)

    # Calculate the TF-IDF for each query term
    for token in query_tokens:
        stemmed_token = stemmer.stem(token)  # Stem each token in the query
        if stemmed_token not in query_weights:
            tf_query = 1 + math.log10(query_tokens.count(token))  # Calculate term frequency in the query
            idf_value = getidf(stemmed_token)  # Get IDF for the token
            query_weights[stemmed_token] = tf_query * idf_value  # Calculate query's TF-IDF weight
            query_length += query_weights[stemmed_token] ** 2  # Sum of squared weights for normalization

    # Normalize the query vector
    query_magnitude = math.sqrt(query_length)
    for token in query_weights:
        query_weights[token] /= query_magnitude  # Divide by the query vector magnitude to normalize

    # Retrieve top-10 documents from postings list for each query term
    top_10_docs = defaultdict(list)  # Stores the top 10 documents for each term
    for token in query_weights:
        if token in postings:
            top_10_docs[token] = postings[token][:10]  # Retrieve top 10 documents based on weight

    # Get documents that appear in the top-10 of all query tokens
    candidate_docs = set(doc_vector.keys())  # Start with all documents
    for token in top_10_docs:
        token_docs = set(doc for doc, _ in top_10_docs[token])  # Get documents for this token
        candidate_docs.intersection_update(token_docs)  # Retain only documents common to all tokens

    # Calculate actual and upper-bound scores
    candidate_scores = Counter()  # Actual scores
    upper_bound_scores = Counter()  # Upper-bound scores

    for token in top_10_docs:
        weight_query = query_weights[token]
        top_docs = top_10_docs[token]

        for doc, doc_weight in top_docs:
            if doc in candidate_docs:
                candidate_scores[doc] += weight_query * doc_weight  # Add actual score

        if len(top_docs) == 10:
            min_weight = top_docs[-1][1]  # Get the lowest weight in the top 10
        else:
            min_weight = 0

        for doc in candidate_docs:
            if doc not in candidate_scores:
                upper_bound_scores[doc] += weight_query * min_weight  # Add to upper-bound score

    # Determine the best document based on the scores
    best_document = None
    best_score = -float('inf')

    for doc in candidate_docs:
        score = candidate_scores[doc]
        total_score = score + upper_bound_scores[doc]  # Consider the total possible score

        if score > best_score:
            best_document = doc
            best_score = score
        elif total_score > best_score:
            best_document = doc
            best_score = total_score

    return best_document, best_score  # Return the best matching document and its score

# Example IDF values
print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))

print("--------------")

# Example weights for specific terms in documents
print("%.12f" % getweight('19_lincoln_1861.txt', 'constitution'))
print("%.12f" % getweight('23_hayes_1877.txt', 'public'))
print("%.12f" % getweight('25_cleveland_1885.txt', 'citizen'))
print("%.12f" % getweight('09_monroe_1821.txt', 'revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt', 'leadership'))

print("--------------")

# Example queries with the optimized search function
print("Best match for 'states laws':", query_with_postings("states laws"))
print("Best match for 'war offenses':", query_with_postings("war offenses"))
print("Best match for 'british war':", query_with_postings("british war"))
print("Best match for 'texas government':", query_with_postings("texas government"))
print("Best match for 'world civilization':", query_with_postings("world civilization"))
