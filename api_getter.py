import requests
import json
import uuid


def api_getter():

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    payload = 'scope=GIGACHAT_API_CORP'  # GIGACHAT_API_PERS
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        # 'Basic MGU0ZmNhZGUtZjhhNS00ZTQ5LTk4MTYtNmZiYTZlOTdmOGQwOjQxYTMzMTc2LWEzZmYtNDI2Yi1iZjY3LTFhNTBkMjQyMTY3YQ=='
        'Authorization': 'Basic YmZlNGMwYWQtM2E0ZS00NzQ3LWIzMzQtZWYxN2NjNTYxODEyOmIyMjkxZTI2LTg4NjktNDc1Yy05NjE5LTg2NzUxNzc3MWZmYg=='
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    token_data = json.loads(response.text)
    access_token = token_data["access_token"]
    expires_at = token_data["expires_at"]
    print(token_data)

    return token_data


