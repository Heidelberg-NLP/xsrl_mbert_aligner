from pre_process.utils_prep import get_bool_value, get_spacy_lang, write_conll_annos, read_sentences
from pre_process.CoNLL_Annotations import CoNLLUP_Token, read_conll
from collections import defaultdict
import argparse
import utils.bert_scorer as bert_scorer
import logging, sys

def alignments_to_file(alignments, filename):
    with open(filename, "w") as f:
        for align_dict in alignments:
            align_pairs = []
            for src, (tgt, _) in sorted(align_dict.items(), key=lambda x: x[0]):
                align_pairs.append(f"{src.split('_')[0]}-{tgt.split('_')[0]}")
            f.write(" ".join(align_pairs)+"\n")


def word_aligner(src_sents, tgt_sents, k, alignment_mode, verbose):
    bert_model, bert_tokenizer = bert_scorer.get_bert_model(model_type="bert-base-multilingual-cased", basic_tokenize=False)
    alignments = []
    for counter, (s, t) in enumerate(zip(src_sents, tgt_sents)):
        logging.info(f"---------- {counter+1} --------------")
        # *** Get the BERT_Token-wise Confusion Matrix ***
        src_bert_tokens, tgt_bert_tokens, bert_sim_matrix = bert_scorer.bert_similarities(s.get_sentence(), t.get_sentence(), 
                                                                                          bert_model, bert_tokenizer)
        
        # *** Get FullWord Alignments ***
        # Make Src-Tgt Dictionary According to the Most BERT-Score similar FullWordPairs. IT IS ZERO-BASED!
        # EXAMPLE: word_pair_dict = {'0_No': ('0_No', 0.9064167), '3_was': ('3_fue', 0.8234986), ...}
        word_pair_dict = bert_scorer.get_most_similar_pairs(s, t, src_bert_tokens, tgt_bert_tokens,
                                                            bert_sim_matrix, get_best=k, alignment_mode="BERT-"+alignment_mode)
        
        alignments.append(word_pair_dict)
        if verbose:
            for item in sorted(word_pair_dict.items(), key=lambda x: x[0]):
                logging.info(item)
    return alignments


if __name__ == "__main__":
    """

    RUN EXAMPLE: 
        python word_aligner.py \
            -s trial_data/CoNLL09_Test_EN_template_test.conll \
            -t trial_data/CoNLL09_Test_ES_template_test.conll \
            --src_lang EN --tgt_lang ES --align_mode INTER

    """

    # Read arguments from command line
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', help='Filepath where the output file will be saved', required=False)
    parser.add_argument('-s', '--src_file', help='Filepath containing the Source Sentences', required=True)
    parser.add_argument('-t', '--tgt_file', help='Filepath containing the Target Sentences', required=True)
    parser.add_argument('-sl', '--src_lang', default="SRC")
    parser.add_argument('-tl', '--tgt_lang', default="TGT")
    parser.add_argument('-v', '--verbose', default="True")
    parser.add_argument('-a', '--align_mode', required=True) # INTER | S2T | T2S
    parser.add_argument('-k', '--k_candidates', default=2, help="The top-most similar word candidates to consider in alignments")
    args = parser.parse_args()

    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"logs/WordAlignment_{args.src_lang}_{args.tgt_lang}_{args.align_mode}.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])

    src_annos = read_conll(args.src_file, conll_token=CoNLLUP_Token, splitter="\t") # Source
    tgt_annos = read_conll(args.tgt_file, conll_token=CoNLLUP_Token, splitter="\t") # Target

    assert len(src_annos) == len(tgt_annos), f"There are {len(src_annos)} sources and {len(tgt_annos)} target sentences"

    word_alignments = word_aligner(src_annos, tgt_annos,
                                    k=args.k_candidates, 
                                    alignment_mode=args.align_mode, 
                                    verbose=get_bool_value(args.verbose))
    
    alignments_to_file(word_alignments, f"{args.src_lang}_{args.tgt_lang}_{args.align_mode}.align")