from langchain.chat_models import GigaChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from api_getter import api_getter
from prompts import MAP_TERM_FIND, DEFINITION_GEN
from document_loaders import text_splitter
import ast
import re


# from tests.tests import TEST_LEC, LEC_2


def glossary(text):
    # Делим на чанки
    chunks = text_splitter(text)

    token = api_getter()["access_token"]
    print(f'Новый токен:\n{token}')
    llm = GigaChat(access_token=token, verify_ssl_certs=False)

    map_term_find = MAP_TERM_FIND
    map_term_find_prompt = PromptTemplate.from_template(map_term_find)
    map_term_find_chain = LLMChain(llm=llm, prompt=map_term_find_prompt)
    all_terms = []
    for chunk in chunks:
        print("INPUT CHUNK:", chunk)
        res = map_term_find_chain.invoke({"chunk": chunk})
        print("OUTPUT:", res["text"])
        all_terms.extend(ast.literal_eval(res["text"]))
    definition_gen = DEFINITION_GEN
    definition_gen_prompt = PromptTemplate.from_template(definition_gen)
    definition_gen_chain = LLMChain(llm=llm, prompt=definition_gen_prompt)
    res = definition_gen_chain.invoke({"terms": str(all_terms)})
    matches = re.findall(r'— (.*?)\.', res['text'])
    print(matches)
    all_matches = []
    for match in matches:
        match.strip()
        all_matches.append(match)
    print("GLOSS:", res["text"])
    full_glossary = []
    for i in range(len(all_matches)):
        pair = {
            "name": all_terms[i],
            "definition": all_matches[i],
            "time": "Soon"
        }
        full_glossary.append(pair)
    print(full_glossary)
    return res

# glossary(TEST_LEC)
