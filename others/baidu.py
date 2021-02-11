import base64
import json
import os
import pathlib

import requests

client_id = ''
client_secret = ''


def get_token(client_id, client_secret):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + client_id + '&client_secret=' + client_secret
    res = requests.get(host, headers=headers).text
    res = json.loads(res)
    return res['access_token']


def get_license_plate(access_token, img_path):
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate?access_token=" + access_token
    with open(img_path, 'rb') as f:
        image_binary = f.read()
    image_encode = base64.b64encode(image_binary)
    postdata = {'image': image_encode, 'multi_detect': 'false'}
    respond = requests.post(url, data=postdata, headers=headers)
    respond.encoding = 'utf-8'
    words_result = json.loads(respond.text)
    if 'words_result' in words_result.keys():
        number = words_result['words_result']['number']
        parent = pathlib.Path(img_path).parent
        dir_name = parent.name
        new_parent = str(parent).replace(dir_name, number)
        os.rename(parent, new_parent)
    else:
        print('%s未识别' % img_path)


if __name__ == '__main__':
    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
    access_token = get_token(client_id, client_secret)

    dir = "/Users/zhaotao/Desktop/data/vanke"
    images = [str(path) + '/img.png' for path in pathlib.Path(dir).glob("*")]

    for img_path in images:
        get_license_plate(access_token, img_path)
