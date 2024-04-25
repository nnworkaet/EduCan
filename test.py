from langchain.chat_models.gigachat import GigaChat
from api_getter import api_getter


token = api_getter()["access_token"]
print(f'Новый токен:\n{token}')
llm = GigaChat(access_token=token, verify_ssl_certs=False, profanity_check=False, verbose=True)
print(llm.invoke(" Курская область и Киев образована 13 июня 1934 года."))