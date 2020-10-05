import argparse
from pre_process.CoNLL_Annotations import read_conll, CoNLLUP_Token, CoNLL09_Token
from pre_process.utils_prep import get_bool_value


def CoNLLUP_to_09(in_filename, out_filename):
    tgt_sentences = read_conll(in_filename, conll_token=CoNLLUP_Token, splitter="\t")
    out_file = open(out_filename, "w")

    for sent in tgt_sentences:
        for token in sent.tokens:
            out_file.write(token.get_conll09_line() + "\n")
        out_file.write("\n")


def CoNLL09_to_UP(in_filename, out_filename, only_verbs):
    tgt_sentences = read_conll(in_filename, conll_token=CoNLL09_Token, splitter="\t")
    out_file = open(out_filename, "w")

    for sent in tgt_sentences:
        out_file.write(sent.get_conllU_anno(only_verbs=only_verbs) + "\n\n")


if __name__ == "__main__":
    """
        RUN EXAMPLE:
        python pre_process/CoNLL_converter.py -s CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_file", help="Source Text File with sentences")
    parser.add_argument("-v", "--only_verbs", help="If true, it only preserves the verbal predicates", default="True") 
    parser.add_argument("-m", "--mode", help="Conversion Type", default="09toUP") # 09toUP | UPto09
    args = parser.parse_args()

    only_verbs = get_bool_value(args.only_verbs)

    if args.mode == "09toUP":
        CoNLL09_to_UP(args.src_file, f"{args.src_file}.conll", only_verbs=only_verbs)
    elif args.mode == "UPto09":
        CoNLLUP_to_09(args.src_file, f"{args.src_file}.conll09")
    else:
        raise NotImplementedError


