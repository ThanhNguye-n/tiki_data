import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_url_image(url):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Fetch the HTML content from the URL
    response = requests.get(url, headers=headers)
    html = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find the <div> tag with class "group-images"
    div_tag = soup.find('div', class_='group-images')

    soup = BeautifulSoup(str(div_tag), 'html.parser')

    img_tag = soup.find('img')
    if img_tag:
        image_url = img_tag['src']
        image_url = image_url.replace("w1200", "w280")

    return image_url


if __name__ == '__main__':

    url = input('Enter an url of product: ')
    product_url = get_url_image(url)

    dict_label = {12: 'OPPO_2647',18: 'Samsung_2647',21: 'Xiaomi_2647',20: 'Vivo_2647',19: 'Tecno_2647',16: 'Realme_2647',8: 'OEM_2647',14: 'Oukitel_2647',
                  6: 'Nokia_2647',0: 'Apple_2647',2: 'Itel_2647',7: 'Nokia_2649',17: 'Samsung_2645',11: 'OEM_2653',3: 'Kindle_2653',13: 'Onyx Boox_2653',
                  4: 'Kobo_2653',9: 'OEM_2649',5: 'Masstel_2649',1: 'Forme_2649',15: 'Panasonic_2651',10: 'OEM_2651',22: 'Yealink_2651'}

    response = requests.get(product_url)
    image = Image.open(BytesIO(response.content))

    # Convert the image to a PyTorch tensor
    img = transforms.functional.to_tensor(image)

    # Initialize the weights
    weights = models.ResNet34_Weights.DEFAULT
    model = torch.jit.load('models/model_ResNet34_25epochs.pt')
    model.eval()

    # Initialize the inference transforms
    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)
    batch = batch.to('cuda')

    # Predict
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)

    class_id = prediction.argmax().item()
    label_name = dict_label[class_id]


    # Recommend product same labels
    df = pd.read_csv('data/full_data_dien_thoai_may_tinh_bang')
    index = df[df.label==label_name].index
    list_products = np.random.choice(index, 3, replace=False)

    print()
    print(f" --> Predict product's label as {dict_label[class_id]}")
    print()
    print('Recommend products have the same label :')

    for idx, i in enumerate(list_products):
        idx +=1
        link = df.loc[i,'thumbnail_url']
        response = requests.get(link)
        print(f'No.{idx} : ')
        image = Image.open(BytesIO(response.content))
        image.show()
        print()