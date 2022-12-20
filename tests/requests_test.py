import requests


def send_api_request(text):
    try:
        url = 'http://{0}:{1}/t2i-person'.format('localhost', 9008)
        return requests.post(url, json={'text': text})
    except requests.exceptions.RequestException as e:
        print(e)


if __name__ == '__main__':

    for i in range(0, 1):
        # health()
        # resp = send_api_request("орк в оранжевой одежде")
        resp = send_api_request("китаец с розовыми волосами")
        resp.raise_for_status()
        print("status:", resp.status_code)
        # print(resp.json)
    # health()
