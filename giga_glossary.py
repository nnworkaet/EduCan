from langchain.chat_models import GigaChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from api_getter import api_getter
from prompts import MAP_TERM_FIND
from document_loaders import text_splitter
from temp.tests import TEST_LEC, LEC_2
import ast


def glossary(text):
    # Делим на чанки
    chunks = text_splitter(text)

    token = api_getter()["access_token"]
    print(f'Новый токен:\n{token}')
    llm = GigaChat(access_token=token, verify_ssl_certs=False)

    map_term_find = MAP_TERM_FIND
    map_term_find_prompt = PromptTemplate.from_template(map_term_find)
    map_term_find_chain = LLMChain(llm=llm, prompt=map_term_find_prompt)
    full_gloss = []
    for chunk in chunks:
        print("INPUT CHUNK:", chunk)
        res = map_term_find_chain.invoke({"chunk": chunk})
        print("OUTPUT:", res["text"])
        if "\n" in res["text"]:
            substrings = res["text"].split("\n")
            final_sets = []
            for substr in substrings:
                pair = ast.literal_eval(substr)
                final_sets.append(pair)
        else:
            final_sets = ast.literal_eval(res["text"])

        print(final_sets)
        for pair in final_sets:
            if len(pair) >= 2:
                part_gloss = {
                    "name": pair[0],
                    "definition": pair[1],
                    "time": "Soon"
                }
                full_gloss.append(part_gloss)

    print("GLOSSARY:", full_gloss)
    return full_gloss

# glossary(TEST_LEC)
