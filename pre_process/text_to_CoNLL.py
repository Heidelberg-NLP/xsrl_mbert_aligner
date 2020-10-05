"""
---
Text --> CoNLLU_Template
---
This file Generates CoNLL Templates from Plain text files (one sentence per line)
"""

import argparse
from pre_process.CoNLL_Annotations import CoNLLUP_Token_Template
import pre_process.utils_prep as utils_prep

if __name__ == "__main__":
    """
        RUN EXAMPLE:
        python pre_process/text_to_CoNLL.py -s trial_data/SentsOnly_ES.txt \
                -o trial_data/ES_template_trial.conll -l ES
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_file", help="Source Text File with sentences")
    parser.add_argument("-o", "--out_file", help="Where to save the CoNLL output file")
    parser.add_argument("-l", "--lang", help="Language for Tokenizing", default="NONE")
    parser.add_argument("-a", "--add_syntax", help="Include Automatic Syntax from SpaCy models", default="False")
    args = parser.parse_args()

    ADD_SYNTAX = utils_prep.get_bool_value(args.add_syntax)
    USE_LARGE_MODEL = False

    spacy_nlp = utils_prep.get_spacy_lang(args.lang, USE_LARGE_MODEL)

    sentences = utils_prep.read_sentences(args.src_file)
    out_file = open(args.out_file, "w")

    tok_sents = []

    for s in sentences:
        spacy_lang = utils_prep.get_spacy_lang(args.lang)
        tok_strs = utils_prep.tokenize_sentence(s, spacy_lang)
        tok_objs = []
        print(tok_strs)
        for i, t in enumerate(tok_strs):
            token = CoNLLUP_Token_Template(i+1, t)
            if ADD_SYNTAX:
                tok_objs.append(token)
            else:
                out_file.write(token.get_conllU_line("\t")+"\n")
        if ADD_SYNTAX:
            tok_sents.append(tok_objs)
        else:
            out_file.write("\n")

    if ADD_SYNTAX:
        for sent in tok_sents:
            anno_tokens = utils_prep.add_dep_info(sent, args.lang, spacy_nlp)
            for t in anno_tokens:
                out_file.write(t.get_conllU_line("\t") + "\n")
            out_file.write("\n")