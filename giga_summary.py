from langchain.chat_models import GigaChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from api_getter import api_getter
from prompts import MAP_SUM, COMBINE_SUM
from document_loaders import text_splitter
import asyncio


# from temp.tests import TEST_LEC

async def summary(text, facts_num):
    # Делим на чанки
    chunks = text_splitter(text)
    print(chunks)

    token = api_getter()["access_token"]
    print(f'Новый токен:\n{token}')
    llm = GigaChat(access_token=token, verify_ssl_certs=False)

    map_sum = MAP_SUM
    map_sum_prompt = PromptTemplate.from_template(map_sum)
    map_chain = LLMChain(llm=llm, prompt=map_sum_prompt)

    # Результат по чанкам
    map_sums_chunks = ""
    for chunk in chunks:
        print("INPUT CHUNK:", chunk)
        sum_part = await asyncio.to_thread(map_chain.invoke, {"chunk": chunk})
        print("OUTPUT RES:", sum_part)
        if sum_part['text'].startswith(("Не люблю менять тему разговора",
                                        "Что-то в вашем вопросе меня смущает",
                                        "Как у нейросетевой языковой модели у меня не может быть настроения")):
            return {"status_code": 401, "Blacklisted chunk": chunk}

        map_sums_chunks += f"{sum_part['text']}\n\n"

    print(map_sums_chunks)
    combine_sum = COMBINE_SUM
    combime_sum_prompt = PromptTemplate.from_template(combine_sum)
    combine_chain = LLMChain(llm=llm, prompt=combime_sum_prompt)
    res = await asyncio.to_thread(combine_chain.invoke, {"map_sums": map_sums_chunks, "facts_num": facts_num})

    return res['text']

# print(asyncio.run(summary(TEST_LEC, 7)))
