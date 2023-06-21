import requests
import pandas as pd
import re
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'vi-VN,vi;q=0.8,en-US;q=0.5,en;q=0.3',
    'x-guest-token': '8jWSuIDBb2NGVzr6hsUZXpkP1FRin7lY',
    'Connection': 'keep-alive',
    'TE': 'Trailers',
}

params = {
    'product_id': None,
    'sort': 'score|desc,id|desc,stars|all',
    'page': '1',
    'limit': '10',
    'include': 'comments'
}

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )
        x = self.drop(output)
        x = self.fc(x)
        return x

def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:return 0
    elif rating == 3:return 1
    else: return 2

def comment_parser(json):
    d = dict()
    d['customer_id']  = json.get('customer_id')
    d['content'] = json.get('content')
    d['rating'] = json.get('rating')
    return d

def predict_sentiment(tokenizer, series):

    checkpoint = torch.load('models/best_phoBert_model.pt', map_location=torch.device(device))
    model = SentimentClassifier(3)
    model.load_state_dict(checkpoint)
    model.eval()

    # Preprocess the input series
    encoding = tokenizer.batch_encode_plus(
        series.tolist(),
        add_special_tokens=True,
        max_length=100,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)

    return pd.Series(predicted.tolist())

if __name__ == "__main__":
    url = input('Enter url of product: ')
    pattern = r"p(\d+)\.html"
    id_match = re.search(pattern, url)
    query_id = int(id_match.group(1))

    params['product_id'] = query_id
    result =[]
    for i in range(2):
        params['page'] = i
        response = requests.get('https://tiki.vn/api/v2/reviews', headers=headers, params=params)
        for comment in response.json().get('data'):
            result.append(comment_parser(comment))

    df_comment = pd.DataFrame(result)
    df_comment.rating = df_comment.rating.apply(to_sentiment)

    predictions = predict_sentiment(tokenizer, df_comment.content)
    print('-'*40)
    print(f'True Label Positive: {len(df_comment[df_comment.rating==2])}/{len(df_comment)}')
    print(f'Predict Label Positive: {len(predictions[predictions==2])}/{len(predictions)}')
    print('-'*40)
    if len(predictions[predictions==2])/len(predictions) > 0.8:
        print('You should buy this product')
    else:
        print('You should not buy this product')
                                          

    

