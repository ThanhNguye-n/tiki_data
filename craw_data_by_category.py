import requests
import json
import pandas as pd
import glob
from pathlib import Path
import concurrent.futures
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Create a session object for persistent requests
session = requests.Session()

# Headers to be used in requests
HEADERS = {
    'authority': 'tiki.vn',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'x-guest-token': 'V1whimCflaEj46Fd3PMxGXW95OHI2rDA',
}

# Keys used to extract data from API response
KEYS = ['id', 'sku', 'name', 'brand_name', 'price', 'discount', 'discount_rate', 'rating_average',
        'review_count', 'order_count', 'favourite_count', 'productset_id', 'seller', 'inventory',
        'stock_item', 'seller_product_id', 'quantity_sold', 'original_price', 'shippable',
        'availability', 'primary_category_path', 'product_reco_score', 'seller_id']

def get_data(data):
    """
    Get data from key 'data'
    """
    rows = []
    if 'data' in data:
        for item in data['data']:
            row = {key: item[key] for key in KEYS}
            rows.append(row)
    return pd.DataFrame(rows)

def get_sub_category(cate_value, max_attempts=5, retry_delay=10):
    """
    Get the sub-categories for a given category value.

    Args:
        cate_value (str): Category value for which sub-categories are to be fetched.
        max_attempts (int): Maximum number of retry attempts in case of request failures.
        retry_delay (int): Delay in seconds before making a retry attempt.

    Returns:
        pandas.DataFrame: DataFrame containing the sub-category data.
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            response = requests.get('https://tiki.vn/api/personalish/v1/blocks/listings',
                                    headers=HEADERS,
                                    params={'category': cate_value, 'aggregations': 2})
            response.raise_for_status()
            data = json.loads(response.text)
            return pd.DataFrame(data['filters'][0]['values'])
        except:
            print("Failed Request ")
            attempts += 1
            if attempts < max_attempts:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retry attempts reached ({max_attempts}).")
    return pd.DataFrame()

def request_func(params, max_attempts=3, retry_delay=10):
    """
    Perform a GET request to the Tiki API and retrieve data based on the provided parameters.

    Args:
        params (dict): Request parameters for the API.
        max_attempts (int): Maximum number of retry attempts in case of request failures.
        retry_delay (int): Delay in seconds before making a retry attempt.

    Returns:
        pandas.DataFrame: DataFrame containing the retrieved data.
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            response = session.get('https://tiki.vn/api/personalish/v1/blocks/listings',
                                   headers=HEADERS,
                                   params=params)
            response.raise_for_status()
            data = json.loads(response.text)
            return get_data(data=data)
        except:
            attempts += 1
            if attempts < max_attempts:
                time.sleep(retry_delay)
            else:
                print(f"WARNING: Max retry attempts reached ({max_attempts}).")
    return pd.DataFrame()


def get_data_from_dataFrame(dataFrame):
    """
    Get data from sub-categories in parallel using multiple threads.
    """

    main_df = pd.DataFrame(columns=KEYS)
    params = {'limit': 100, 'aggregations': 2}
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for cate in range(0, dataFrame.shape[0]):
            for page in range(1, 21):
                params['category'] = dataFrame.loc[cate, 'query_value']
                params['page'] = str(page)
                futures.append(executor.submit(request_func, params.copy()))

    for future in concurrent.futures.as_completed(futures):
        main_df = main_df.append(future.result(), ignore_index=True)

    return main_df


def get_data_apply_filter(cate_value):
    """
    Get data by applying filters for a specific category.
    """

    main_df = pd.DataFrame(columns=KEYS)
    params = {'limit': 100, 'aggregations': 2}
    price = [(0, 50000), (50001, 80000), (80001, 110000), (110001, 140000),
             (140001, 170000), (170001, 200000), (200001, 230000), (230001, 260000),
             (260001, 300000), (300001, 350000), (350001, 400000), (400000, 600000),
             (600001, 100000000)]

    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for filter_range in price:
            for page in range(1, 21):
                params['category'] = cate_value
                params['page'] = str(page)
                params['price'] = str(filter_range[0]) + ',' + str(filter_range[1])
                futures.append(executor.submit(request_func, params.copy()))

    for future in concurrent.futures.as_completed(futures):
        main_df = main_df.append(future.result(), ignore_index=True)

    return main_df

def crawling_data_tiki(CATEGORY_VALUE):
    """
    Perform the crawling of data from Tiki for a specific category.

    Args:
        CATEGORY_VALUE (str): Category value for which data is to be crawled.

    Returns:
        pandas.DataFrame: DataFrame containing the crawled data.
    """

    df = get_sub_category(cate_value=CATEGORY_VALUE)
    main_df = pd.DataFrame(columns=KEYS)

    with tqdm(total=df.shape[0], desc="Progress") as pbar:
        start_time = time.time()
        for idx in range(0, df.shape[0]):
            df_sub = get_sub_category(cate_value=df.loc[idx, 'query_value'])

            if 'url_key' not in df_sub.columns:
                data_sub = get_data_apply_filter(cate_value=df.loc[idx, 'query_value'])
                main_df = main_df.append(data_sub, ignore_index=True)
            else:
                small_df = df_sub[df_sub['count'] <= 2000].reset_index(drop=True)
                huge_df = df_sub[df_sub['count'] > 2000].reset_index(drop=True)

                for i_sub in range(0, huge_df.shape[0]):
                    data_sub = get_data_apply_filter(cate_value=huge_df.loc[i_sub, 'query_value'])
                    main_df = main_df.append(data_sub, ignore_index=True)

                temp_df = get_data_from_dataFrame(small_df)
                main_df = main_df.append(data_sub, ignore_index=True)

            elapsed_time = time.time() - start_time
            pbar.set_postfix({"Elapsed Time": elapsed_time})
            pbar.update(1)

    main_df['quantity_sold'] = main_df['quantity_sold'].str.get('value')
    main_df.drop_duplicates().reset_index(drop=True, inplace=True)

    return main_df


if __name__ == '__main__':
    user_input = input("Enter a category name: ")
    user_input = "-".join(user_input.strip().split())

    with open('category_id.json') as file:
        data = json.load(file)

    for item in data:
        if item['name'] == user_input:
            value = item['value']
            break
        else:
            value = None

    if value is not None:
        print('Crawling data is in progress...')
        df_crawled = crawling_data_tiki(CATEGORY_VALUE=value)
        df_crawled.to_csv(f'{user_input}.csv', index=False)
        print(f'Save file as {user_input}.csv')
    else:
        print('Category name is not found.')


