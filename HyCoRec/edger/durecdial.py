import json
from tqdm import tqdm

DATA_ROOT = "data/dataset/durecdial/jieba"

def get_side_data():
    side_data = []
    with open(f"{DATA_ROOT}/entity_subkg.txt", "r", encoding="utf-8") as file:
        for line in file.readlines():
            [a, _, b] = line.split("\t")
            b = b[:-1] if b[-1] == "\n" else b
            side_data.append((a, b))

    entity_side = {}
    for a, b in tqdm(side_data):
        if a in entity_side:
            entity_side[a].add(b)
        else:
            entity_side[a] = set([b])
        if b in entity_side:
            entity_side[b].add(a)
        else:
            entity_side[b] = set([a])
    for a in entity_side:
        b = entity_side[a]
        entity_side[a] = list(b)
    
    word_side = {}
    token_set = set([token.lower() for token in json.load(open(f"{DATA_ROOT}/token2id.json", "r", encoding="utf-8"))])
    with(open("data/conceptnet/zh_side.txt", "r", encoding="utf-8")) as concept_net_words:
        for words in tqdm(concept_net_words.readlines()):
            a, b = words[:-1].split(" ")
            if a not in token_set or b not in token_set:
                continue
            if a in word_side:
                word_side[a].add(b)
            else:
                word_side[a] = set([b])
            if b in word_side:
                word_side[b].add(a)
            else:
                word_side[b] = set([a])
        for a in word_side:
            b = word_side[a]
            word_side[a] = list(b)

    return entity_side, word_side

def durecdial_edger():
    item_edger, word_edger = get_side_data()
    entity_edger = item_edger
    return item_edger, entity_edger, word_edger