import requests
import json


DRESSNET_POST = 'http://deepnets.nyc/imageraw/'
DRESSNET_GET = 'http://deepnets.nyc/predict/'


def post_image(filename):
    files = {'file': ('dont_care', open(filename, 'rb'), 'application/x-www-form-urlencoded')}

    response = requests.post(DRESSNET_POST, files=files)

    if response.status_code == 200:
        return json.loads(response.text)['file_id']
    else:
        return None


def predict(file_id, top_n=None):
    params = {'file_id': file_id}
    if top_n is not None:
        params['top_n'] = top_n

    response = requests.get(DRESSNET_GET, params={'file_id': file_id, 'top_n': 10})

    if response.status_code == 200:
        return json.loads(response.content)
    else:
        return None


def example():
    file_id = post_image('/Users/jade/work/Eigenstyle/cust_im/06_507186.jpg')
    predictions = predict(file_id, top_n=3)
    return predictions
