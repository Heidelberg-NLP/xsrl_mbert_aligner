import spacy

def read_sentences(filepath):
    with open(filepath) as f:
        for line in f.readlines():
            yield line.strip()


def get_bool_value(str_bool):
    if str_bool.upper() == "TRUE" or str_bool.upper() == "T":
        return True
    else:
        return False


def tokenize_sentence(sentence_str, spacy_lang=None):
    if not spacy_lang:
        return sentence_str.split()
    else:
        return [tok.text for tok in spacy_lang.tokenizer(sentence_str)]


def add_dep_info(tgt_tokens, lang, spacy_nlp, include_detail_tag=True):
    """
    :param tgt_tokens: a list of CoNLLUP_Token_Template() Objects from CoNLL_Annotations.py file
    :param spacy_nlp: Spacy language model of the target sentence to get the proper Dependency Tree
    :return:
    """
    doc = spacy_nlp.tokenizer.tokens_from_list([t.word for t in tgt_tokens])
    spacy_nlp.tagger(doc)
    spacy_nlp.parser(doc)
    for ix, token in enumerate(doc):
        tgt_tokens[ix].lemma = token.lemma_ or "_"
        tgt_tokens[ix].head = token.head.i + 1
        if lang in ["ES", "FR"]:
            detail_tag = token.tag_.split("__") # [VERB , Mood=Sub|Number=Plur|Person=3|Tense=Imp|VerbForm=Fin]
            tgt_tokens[ix].pos_tag = detail_tag[0] or "_"
            if include_detail_tag:
                tgt_tokens[ix].detail_tag = detail_tag[-1] or "_"
        else:
            tgt_tokens[ix].pos_tag = token.tag_ or "_"
        tgt_tokens[ix].pos_universal = token.pos_ or "_" # Is SpaCy already Universal?
        tgt_tokens[ix].dep_tag = token.dep_ or "_"
        tgt_tokens[ix].ancestors = [(t.i, t.text) for t in token.ancestors]
        tgt_tokens[ix].children = [(t.i, t.text) for t in token.children]

        # print(token.i, token.text, token.pos_, token.dep_, token.head.text, token.head.i, token.tag_)
    assert len(doc) == len(tgt_tokens), f"LEN Mismatch! Spacy has {len(doc)} tokens and CoNLL has {len(tgt_tokens)} tokens"
    return tgt_tokens


def get_spacy_lang(lang, use_large_model=False):
    if lang =="EN":
        if use_large_model:
            return spacy.load("en_core_web_lg")
        else:
            return spacy.load("en")
    elif lang =="ES":
        if use_large_model:
            return spacy.load("es_core_news_md")
        else:
            return spacy.load("es")
    elif lang == "DE":
        if use_large_model:
            return spacy.load("de_core_news_md")
        else:
            return spacy.load("de_core_news_sm")
    elif lang == "FR":
        if use_large_model:
            return spacy.load("fr_core_news_md")
        else:
            return spacy.load("fr")
    else:
        return None


def write_conll_annos(annos, outname):
    with open(outname, "w") as f:
        for anno in annos:
            for token in anno.tokens:
                f.write(token.get_conllU_line()+"\n")
            f.write("\n")