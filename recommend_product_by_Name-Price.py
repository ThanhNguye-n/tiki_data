import re
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


def preprocess_text(text):
    """
    Preprocesses the input text by removing HTML tags, punctuation, and converting it to lowercase.
    """
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags (if any)
    text = re.sub(r'[,.:;"“”\'!?-]', ' ', text)  # remove punctuation
    text = text.lower()

    tokens = word_tokenize(text, format="text")  # tokenize text using underthesea

    return tokens


def get_word_embedding(dataset, name_product_query):
    """
    Generates GloVe embeddings for the product names in the dataset.
    Computes the query embedding for the given product name.
    """
    # Preprocess the product names in the dataset
    dataset['preprocess_name'] = dataset['name'].apply(preprocess_text)

    product_names = dataset.preprocess_name.to_list()
    prices = dataset.price.to_list()

    glove_file = 'data/glove.6B.50d.txt'
    glove_word_vectors = {}

    # Load pre-trained GloVe word vectors
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_word_vectors[word] = vector

    def get_glove_embedding(product_name):
        words = product_name.lower().split()
        # Get embeddings for words present in GloVe word vectors
        embeddings = [glove_word_vectors[word] for word in words if word in glove_word_vectors]
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None

    glove_embeddings = []
    # Generate GloVe embeddings for each product name in the dataset
    for product_name in product_names:
        glove_embedding = get_glove_embedding(product_name)
        if glove_embedding is not None:
            glove_embeddings.append(glove_embedding)

    glove_embeddings = np.array(glove_embeddings)
    # Combine the GloVe embeddings with prices to create the combined data
    combined_data = np.column_stack((glove_embeddings, np.array(prices).reshape(-1, 1)))
    # Compute the query embedding for the given product name
    query_embedding = get_glove_embedding(name_product_query)

    return product_names, prices, glove_embeddings, combined_data, query_embedding



def get_list_recommended_product(data, NAME, PRICE):
    """
    Generates a list of recommended products based on the query product.
    Filters products based on clustering and price range.
    Returns the list of recommended products.
    """
    product_names, prices, glove_embeddings, combined_data, query_embedding = get_word_embedding(data, NAME)

    # Apply DBSCAN clustering algorithm to the combined data
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(combined_data)

    query_product = NAME
    query_price = PRICE
    price_range = query_price / 4  # Define the price range for recommended products

    # Compute pairwise cosine similarity between the query embedding and glove embeddings
    query_scores = cosine_similarity([query_embedding], glove_embeddings)[0]

    # Find the cluster label with the highest cosine similarity score
    query_label = labels[np.where(query_scores == np.max(query_scores))][0]

    filtered_products = []
    for product_name, price, label in zip(product_names, prices, labels):
        # Filter products based on the query label and price range
        if label == query_label and abs(price - query_price) <= price_range:
            filtered_products.append((product_name, price))

    return filtered_products


if __name__ == "__main__":
    data = pd.read_csv('https://raw.githubusercontent.com/ThanhNguye-n/tiki_data/main/data/dien-thoai-may-tinh-bang.csv')
    data.drop_duplicates(inplace=True)

    url = input('Enter url of product: ')
    pattern = r"p(\d+)\.html"
    id_match = re.search(pattern, url)
    query_id = int(id_match.group(1))

    # Get name and price of query product
    name = data[data.id == query_id].name.values[0]
    price = data[data.id == query_id].price.values[0]

    # Build recommendation system
    list_products = get_list_recommended_product(data, name, price)
    print(f"Recommended products for: {name} priced at {price} VNĐ\n")

    for idx, (product, price) in enumerate(list_products[:5]):
        idx += 1
        print(f'{idx}. Product: {product}\n   Price: {price} VNĐ')
        print()
