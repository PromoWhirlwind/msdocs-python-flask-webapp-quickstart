import pymongo
import chromadb

import gdown
import os

url = 'https://drive.google.com/uc?export=download&id=1R5WLyiu4LxCve2NbMpqvax2uSr9jGuwR'
output = 'chroma.sqlite3'

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)
else:
    print(f"{output} already exists. Skipping download.")


# Connect to MongoDB
def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri, appname="devrel.content.python")
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

mongo_uri = "mongodb://mongodbdev:mongodbdev@mongodb.centdevelop.mq.ai:27017/?authMechanism=SCRAM-SHA-1&authSource=mongodb&directConnection=true"
mongo_client = get_mongo_client(mongo_uri)

# Define MongoDB database and collection names
DB_NAME = "mongodb"
COLLECTION_NAME = "temp_static_amr_data"
db = mongo_client[DB_NAME]
MongoDB_collection = db[COLLECTION_NAME]

document_count = MongoDB_collection.count_documents({})
print(f"Total Number of Documents in {COLLECTION_NAME}: {document_count}")

full_collection_data = list(MongoDB_collection.find({}, {'Data.Title': 1, 'ResolvableItems': 1, '_id': 1}))


import os
from tqdm import tqdm
import torch
from openai import AzureOpenAI

#os.environ['AZURE_OPENAI_API_KEY'] = 'KEY_HERE'
os.environ['AZURE_OPENAI_API_KEY'] = '360f5fb127c949c1925dde6ea394dc87'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://mq-develop-openai.openai.azure.com/'
os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = 'mq-develop-gpt-4o-2024-05-13-model' #GPT4o
#os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = 'mq-develop-gpt-35-turbo-0301-model' #GPT3.5


Azureclient = AzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
)

chroma_client = chromadb.PersistentClient(path="./")
collection = chroma_client.get_or_create_collection("MongoDB")


import numpy as np
import re

def generate_response(client, question):
    
    # Generate a simplified query using GPT-4o
    prompt = f"Given the following question, generate an expanded query, providing only the expanded query you generated. Include wording you expect to find in the article. Please keep the added context relatively short and simple. You can filter for companies at the end by surrounding the list of lowercase basic company names with [[double brackets]] and using $and or $or to logically seperate multiple companies. Please do not just generally filter for companies and instead use companies mentioned in the user's question. This response will be used with embeddings to find the most relevant article(s) to help answer the question:\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": prompt}],
    #    temperature=0.0,
    )
    
    response_text = response.choices[0].message.content.strip()
    # Extract company names and separate them from the response
    import re
    company_names = re.findall(r"\[\[(.*?)\]\]", response_text)
    company_names = ', '.join(company_names) if company_names else ''
    response_text = re.sub(r"\[\[(.*?)\]\]", "", response_text).strip()
    
    return response_text, company_names


def answer_question(client, question, fulltext):
    # Construct a response based on relevant articles
    print("GPT ANSWER")
    prompt = f"""Directions:
    '''Using the following article(s) information to help answer the query, provide a concise answer to the question, providing direct quotes to the answer in the article. 
    Do NOT make up or hallucinate information, instead using the articles themselves to both come up with and ground your response. Try not to round numbers unless specified in the prompt.
    When responding, make sure to be precise with your wording and terminology. Do not make terms more vague as that can cause confusion. Multiple sources can be used together to help find an answer.
    Please reference the article IDs using exact quotes following MLA format that include helpful information in your answer as citations (if you cite please include direct quotes with \"...\" for gaps and [bracketed text] for intergections or replaced words).
    After finishing creating your anwer, please fact check it with the provided articles to ensure it is correct. Let the user know if the article does not give enough information to answer or infer an answer for the user's question.
    '''
    
    Question: {question} 
    
    Articles:\n\n{fulltext}"""
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    answer = response.choices[0].message.content.strip()
    return answer, prompt

def get_surrounding_ids(document_id, collection, span=1):

    # Extract the numeric part of the document ID assumed to be after the last underscore
    base_id = document_id.rsplit('_', 1)[0]
    doc_number = int(document_id.rsplit('_', 1)[1])
    
    # Generate IDs for the surrounding documents
    surrounding_ids = [f"{base_id}_{str(doc_number + i).zfill(4)}" for i in range(-span, span + 1) if i != 0]
    #print(surrounding_ids)
    # Include the central document ID
    surrounding_ids.insert(span, document_id)
    
    # Retrieve documents
    documents = []
    document_ids = []
    combined_page_content = []
    for doc_id in surrounding_ids:
        response = collection.get(ids=[doc_id])
        if response and response['documents']:
            documents.append(response['documents'][0])
            document_ids.append(response['ids'][0])
            combined_page_content.append(response['documents'][0])

    # Combine page contents into a single string
    combined_page_content = " ".join(combined_page_content)
    
    return combined_page_content

def filter_documents_by_query(full_collection_data, query_string):

    # Function to recursively evaluate the logical expressions
    def evaluate_expression(doc, expression):
        # Base case: simple expression without any logical operators
        if '$and' not in expression and '$or' not in expression:
            relevant_names = [item['Name'].lower() for item in doc['ResolvableItems'] if item['Tag'] in ['Company', 'Contributor', 'Author']]
            expression = expression.strip().lower()
            return any(expression in name for name in relevant_names)

        # Recursive case: expression with logical operators
        if '$or' in expression:
            parts = re.split(r' \$or ', expression)
            return any(evaluate_expression(doc, part) for part in parts)
        elif '$and' in expression:
            parts = re.split(r' \$and ', expression)
            return all(evaluate_expression(doc, part) for part in parts)
        return False

    # Filter documents based on the complex query
    filtered_documents = [doc for doc in full_collection_data if evaluate_expression(doc, query_string)]
    document_ids = [doc['_id'] for doc in filtered_documents]
    document_titles = [doc['Data']['Title'] for doc in filtered_documents]
    return document_ids, document_titles


def get_most_relevant_articles_slow(collection, query_text, n_results=3, filter_ids=[]):
    # Method to retrieve the most relevant documents based on query embeddings
    # Optionally filters results to include only specific articles if filter_ids is not empty
    # Optionally filters results to include only specific articles if filter_ids is not empty
    if filter_ids:
        where_clause = {"BaseID": {"$in": filter_ids}}
    else:
        where_clause = {}
    results = collection.query(
        query_texts=[query_text],  # Chroma will embed this for you
        n_results=n_results,  # number of results to return
        where=where_clause
    )
    # Extracting data from the results dictionary
    ids = results['ids'][0] if 'ids' in results else []
    distances = results['distances'][0] if 'distances' in results else []
    documents = results['documents'][0] if 'documents' in results else []
    
    return ids, documents, distances

def get_most_relevant_articles(collection, query_text, expanded_query_text = "", n_results=3, filter_ids=[]):
    # Fetch the top 1000 results without filtering by filter_ids
    results = collection.query(
        query_texts=[query_text, expanded_query_text],  # Chroma will embed this for you
        n_results=1000  # Fetch top 1000 results
    )
    # Extracting data from the results dictionary
    all_ids = results['ids'][0] if 'ids' in results else []
    all_distances = results['distances'][0] if 'distances' in results else []
    all_documents = results['documents'][0] if 'documents' in results else []

    # Apply filtering after fetching the results if filter_ids is not empty
    if filter_ids:
        # Adjust the filtering to include IDs that start with any of the filter_ids followed by '_xxxx'
        filtered_results = [(id, doc, dist) for id, doc, dist in zip(all_ids, all_documents, all_distances)
                            if any(id.startswith(f"{filter_id}_") for filter_id in filter_ids)]
        # Sort the filtered results by distance to get the most relevant results
        filtered_results.sort(key=lambda x: x[2])
        # Select the top n_results from the sorted list
        ids, documents, distances = zip(*filtered_results[:n_results]) if filtered_results else ([], [], [])
    else:
        # If no filter_ids provided, simply return the top n_results based on distance
        sorted_results = sorted(zip(all_ids, all_documents, all_distances), key=lambda x: x[2])
        ids, documents, distances = zip(*sorted_results[:n_results]) if sorted_results else ([], [], [])

    return ids, documents, distances

def RAG(query, Azureclient, collection, full_collection_data):
    print(query)
    # Generate an expanded query using LLM before finding relevant articles
    expanded_query, companies_string = generate_response(Azureclient, query)
    #print(expanded_query)
    print("Restricting articles to: " + companies_string)
    #print(companies_string)
    IDs, Titles = filter_documents_by_query(full_collection_data, companies_string)
    #If IDs are not together, get all that are seperate
    if companies_string and not IDs:
        print("Catch: Seperating Companies")
        companies_list = companies_string.split(", ")
        for company in companies_list:
            temp_IDs, temp_Titles = filter_documents_by_query(full_collection_data, company)
            IDs.extend(temp_IDs)
            Titles.extend(temp_Titles)
    print(IDs)
        # Find relevant articles based on the expanded query
    long_ids, chunked_texts, distances = get_most_relevant_articles(collection, query, expanded_query, 3, IDs)
    short_ids = []
    for long_id in long_ids:
        if re.search(r'_\d{4}$', long_id):
            short_id = re.sub(r'_\d{4}$', '', long_id)
            short_ids.append(short_id)

    combined_data = list(zip(long_ids, chunked_texts, distances, short_ids))
    all_data = []
    for article in combined_data:
        article_id = article[0]
        short_id = article[3]
        combined_texts = get_surrounding_ids(article_id, collection, 7)
        #print(combined_texts)
        # Combine all texts for this short ID
        full_text = "".join(combined_texts)
        article_title = next((title for id, title in zip(IDs, Titles) if id == short_id), "Title Not Available")

        # Extracting authors and formatting names as "First Name Last Name"
        authors_list = [author['Name'] for item in full_collection_data if item.get('_id') == short_id for author in item.get('ResolvableItems', []) if author.get('Tag') == 'Author']
        formatted_authors = [f"{parts[1]} {parts[0]}" if len((parts := author.split(', '))) == 2 else author for author in authors_list]
        authors_str = ", ".join(formatted_authors)

        # Extracting contributors
        contributors_list = [contributor['Name'] for item in full_collection_data if item.get('_id') == short_id for contributor in item.get('ResolvableItems', []) if contributor.get('Tag') == 'Contributor']
        contributors_str = ", ".join(contributors_list)

        # Combine all information into a single string
        combined_info = f"Article Reference ID {short_id}\nArticle Title: \"{article_title}\"\nAuthors: {authors_str}\nContributors: {contributors_str}\nSegment of Article's Text:\n{full_text}"
        all_data.append(combined_info)

    # Sort all data by the first occurrence of each short ID in the original list
    sorted_all_data = sorted(all_data, key=lambda x: short_ids.index(re.search(r'Article Reference ID (\S+)', x).group(1)))
    full_combined_text = "\n\n".join(sorted_all_data)

    full_combined_text = "\n\n".join(all_data)
    # Ensure that the data passed to answer_question is a list of dictionaries as expected
    #articles_data = [{"document": article_id, "text": text} for article_id, text in zip(long_ids, chunked_texts)]
    answer, prompt = answer_question(Azureclient, query, full_combined_text)
    i = 0
    first_reference_added = False
    added_references = set()
    while i < len(IDs):
        # Append each ID and article title pair on a new line after two new lines
        if str(IDs[i]) in answer and IDs[i] not in added_references:
            if not first_reference_added:
                answer += "\n\nReferences:"
                first_reference_added = True
            answer += f"\n{IDs[i]}: \"{Titles[i]}\""
            added_references.add(IDs[i])
        i += 1
    return answer, prompt













from flask import Flask, request, jsonify, render_template
import threading
import os
import signal
import socket

app = Flask(__name__)

def website_respond(user_input):
    # Replace these function calls with your actual implementation
    answer, prompt = RAG(user_input, Azureclient, collection, full_collection_data)
    fulltext = prompt
    print("Processing complete")  # Add this line
    return answer, fulltext

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/respond', methods=['POST'])
def respond():
    data = request.json
    user_input = data.get("input", "")
    answer, fulltext = website_respond(user_input)
    return jsonify({"response": answer, "fulltext": fulltext})

def run_app():
    port = 5601
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    if result == 0:
        print(f"Port {port} is already in use. Trying to free it.")
        os.system(f"fuser -k {port}/tcp")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Stopping the server.')
    os.kill(os.getpid(), signal.SIGTERM)

signal.signal(signal.SIGINT, signal_handler)

thread = threading.Thread(target=run_app)
thread.start()

thread = threading.Thread(target=run_app)
thread.start()
