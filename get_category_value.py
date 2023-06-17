import requests
import json
import re


headers = {
    'authority': 'tiki.vn',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'x-guest-token': 'V1whimCflaEj46Fd3PMxGXW95OHI2rDA',
}

#request open api
response = requests.get('https://api.tiki.vn/raiden/v2/menu-config', headers=headers, params={'platform':'desktop'})
data = json.loads(response.text)

#get a url link, ex: 'https://tiki.vn/do-choi-me-be/c2549'
urls = [i['link'] for i in data['menu_block']['items']]

#get a name and value using re, name='do-choi-em-be, value=2549
name_pattern = r"\/([a-z0-9-]+)\/"
value_pattern = r"c(\d+)"


data = []

for url in urls:
    name_match = re.search(name_pattern, url)
    name = name_match.group(1) if name_match else None

    value_match = re.search(value_pattern, url)
    value = value_match.group(1) if value_match else None

    url_data = {
        "name": name,
        "value": value
    }

    data.append(url_data)

# Save data as JSON file
with open("category_id.json", "w") as file:
    json.dump(data, file, indent=4)

print("Data saved to category_id.json")