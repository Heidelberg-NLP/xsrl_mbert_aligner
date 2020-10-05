from pre_process.CoNLL_Annotations import CoNLL09_Token, CoNLLUP_Token, ZAPToken, read_conll
from pre_process.utils_prep import get_bool_value, get_spacy_lang, write_conll_annos
from collections import defaultdict
import argparse
import utils.bert_scorer as bert_scorer
import logging, sys
import utils.utils_eval as ue


def populate_conll_template(pred_arg_struct, word_to_tags, token_templates):
    # Make Predicates Easy to Retrieve
    pred_dict = {}
    print(pred_arg_struct.keys())
    for pred_ix, _, pred_sense in pred_arg_struct.keys():
        pred_dict[int(pred_ix)] = pred_sense
    assert len(pred_dict) == len(pred_arg_struct), f"Pred-Arg Struct is Malformed: \n {pred_arg_struct}"
    # Iterate tokens and Add Info...
    for itok in range(len(token_templates)):
        # Populate Predicate Info
        if itok in pred_dict:
            isense = pred_dict[itok]
            token_templates[itok].is_pred = True
            token_templates[itok].pred_sense = isense
            token_templates[itok].pred_sense_id = str(itok) + "##" + isense
        # Populate Arguments
        labels = ["_"] * len(pred_arg_struct)
        for tag in word_to_tags.get(itok, []):
            labels[tag[0]] = tag[1]
        token_templates[itok].labels = labels
    return token_templates

def get_valid_indices(filename):
    valid_ids, qualities = [], []
    with open(filename) as f:
        for line in f.readlines():
            sent_id, q, _ = line.strip().split("\t")
            valid_ids.append(int(sent_id))
            qualities.append(int(q))
    return valid_ids, qualities

def filter_annos(src_annos, valid_ids):
    valid_annos = []
    for i in range(len(src_annos)):
        if i in valid_ids:
            valid_annos.append(src_annos[i])
    return valid_annos

# # ---------------------------

def _get_aligned_word(src_word, tag, mapping):
    w_align = mapping.get(src_word, None)
    if w_align:
        tgt_ix, tgt_word = w_align[0].split("_")
        return int(tgt_ix), tgt_word, tag
    else:
        return None


def get_bertscore_mapping(src_arg_struct, word_pair_dict):
    # --- Transfer What we Find in The SRC->TGT Dict to the TGT Words
    logging.info(word_pair_dict)
    bert_pred_arg_struct = {}
    predicates = []
    word_to_tags = defaultdict(list) # Index of target word mapped to the Tags that it has
    found_preds = 0
    duplicates = []
    for predicate, arguments in src_arg_struct:
        src_ix, src_w, src_sense, _ = predicate
        tgt_pred = _get_aligned_word(f"{src_ix}_{src_w}", src_sense, word_pair_dict)
        if tgt_pred and tgt_pred[0] not in duplicates:
            tgt_pred_ix, tgt_pred_word, tgt_pred_sense = tgt_pred
            predicates.append(f"{tgt_pred_ix}_{tgt_pred_word}_{tgt_pred_sense}")
            duplicates.append(tgt_pred_ix)
            bert_pred_arg_struct[(int(tgt_pred_ix), tgt_pred_word, tgt_pred_sense)] = []
            for arg in arguments:
                src_arg_ix, _, src_arg_tag, src_arg_head = arg.get()
                tgt_arg = _get_aligned_word(f"{src_arg_ix}_{src_arg_head}", src_arg_tag, word_pair_dict)
                if tgt_arg:
                    bert_pred_arg_struct[(int(tgt_pred_ix), tgt_pred_word, tgt_pred_sense)].append(tgt_arg)
                    word_to_tags[int(tgt_arg[0])].append((found_preds, tgt_arg[-1]))
            found_preds += 1

    # We return the predicates and the Arg-Struct of the TARGET SIDE
    return predicates, bert_pred_arg_struct, word_to_tags


def _correct_arg_head(src_arg_ix, src_arg_head, tgt_conll_tokens):
    # 1) Special Case NE companies:
    new_ix = src_arg_ix + 1
    if new_ix < len(tgt_conll_tokens):
        next_tok = tgt_conll_tokens[new_ix].word
        if next_tok in ["Inc", "Ltd", "Co", "Corp", "Inc.", "Ltd.", "Co.", "Corp."]:
            return new_ix, next_tok
    return src_arg_ix, src_arg_head


def get_special_mapping(src_arg_struct, tgt_conll_tokens, word_pair_dict, lang):
    """
    Idea is to Use BERT Cosine Similarity + Check for POS Tags to filter predicates
    :param src_arg_struct:
    :param tgt_conll_tokens:
    :param word_pair_dict:
    :return:
    """
    # --- Transfer What we Find in The SRC->TGT Dict to the TGT Words
    # PREDICATE = (6, 'negaron', 'refuse.01', 'VERB')
    # ARGS [(7, (6, 'negaron', 'refuse.01', 'VERB'), 'A1', 'a'), ..., ...]
    # logging.info(word_pair_dict)
    pred_arg_struct = {}
    predicates = []
    word_to_tags = defaultdict(list) # Index of target word mapped to the Tags that it has
    found_preds = 0

    [print(x) for x in word_pair_dict.items()]
    for predicate, arguments in src_arg_struct:
        src_ix, src_w, src_sense, _ = predicate # _ = src_tag
        tgt_pred = _get_aligned_word(f"{src_ix}_{src_w}", src_sense, word_pair_dict)
        if tgt_pred:
            tgt_pred_ix, tgt_pred_word, tgt_pred_sense = tgt_pred
            pred_str = f"{tgt_pred_ix}_{tgt_pred_word}_{tgt_pred_sense}"
            if tgt_conll_tokens[tgt_pred_ix].pos_universal in ["VERB", "AUX"] and pred_str not in predicates:
                predicates.append(pred_str)
                pred_arg_struct[(int(tgt_pred_ix), tgt_pred_word, tgt_pred_sense)] = []
                for arg in arguments:
                    src_arg_ix, _, src_arg_tag, src_arg_head = arg.get() # _ = src_pred_tuple
                    src_arg_ix, src_arg_head = _correct_arg_head(src_arg_ix, src_arg_head, tgt_conll_tokens)
                    tgt_arg = _get_aligned_word(f"{src_arg_ix}_{src_arg_head}", src_arg_tag, word_pair_dict)
                    if tgt_arg: #  and "C-" not in src_arg_tag
                        pred_arg_struct[(int(tgt_pred_ix), tgt_pred_word, tgt_pred_sense)].append(tgt_arg)
                        word_to_tags[int(tgt_arg[0])].append((found_preds, tgt_arg[-1]))
                found_preds += 1
            else:
                logging.debug(f" SKEPT! {tgt_pred_ix} {tgt_pred_word} {tgt_pred_sense}")

    # We return the predicates and the Arg-Struct of the TARGET SIDE
    return predicates, pred_arg_struct, word_to_tags


def transfer_annotations(src_annos, tgt_annos, lang, alignment_mode, k, compare_gold=False):
    all_bert_preds = []
    new_annos = []

    bert_model, bert_tokenizer = bert_scorer.get_bert_model(model_type="bert-base-multilingual-cased", basic_tokenize=False)

    if compare_gold:
        pred_excess, pred_missed, pred_match = defaultdict(int), defaultdict(int), defaultdict(int)
        arg_excess, arg_missed, arg_match = defaultdict(int), defaultdict(int), defaultdict(int)
        all_excess, all_missed, all_match = defaultdict(int), defaultdict(int), defaultdict(int)

    for counter, (s, t) in enumerate(zip(src_annos, tgt_annos)):
        logging.info(f"---------- {counter+1} --------------")
        # *** Get the BERT_Token-wise Confusion Matrix ***
        src_bert_tokens, tgt_bert_tokens, bert_sim_matrix = bert_scorer.bert_similarities(s.get_sentence(),
                                                                                          t.get_sentence(),
                                                                                          bert_model,
                                                                                          bert_tokenizer)
        
        # *** Get FullWord Alignments ***
        # Make Src-Tgt Dictionary According to the Most BERT-Score similar FullWordPairs. IT IS ZERO-BASED!
        # EXAMPLE: word_pair_dict = {'0_No': ('0_No', 0.9064167), '3_was': ('3_fue', 0.8234986), ...}
        word_pair_dict = bert_scorer.get_most_similar_pairs(s, t, src_bert_tokens, tgt_bert_tokens,
                                                            bert_sim_matrix, get_best=k, alignment_mode=alignment_mode)

        # *** Transfer Labels Using Alignments ***
        # EXAMPLE: bert_tgt_preds = ['2_circuito_install.01', '12_fallaron_fail.01', '18_dicen_say.01']
        # EXAMPLE: word_to_tags   = {4: [(0, 'AM-TMP')], 19: [(1, 'AM-PRD')], 12: [(2, 'A1')], 18: [(2, 'A0')]}
        # Map the Predicates ...
        if "BERT" in alignment_mode:
            bert_tgt_preds, bert_tgt_pred_arg_struct, word_to_tags = get_bertscore_mapping(s.argument_structure.items(), word_pair_dict)
        else:
            bert_tgt_preds, bert_tgt_pred_arg_struct, word_to_tags = get_special_mapping(s.argument_structure.items(),
                                                                                         t.tokens,
                                                                                         word_pair_dict,
                                                                                         lang)

        logging.info(f"\n----------\n{t.get_sentence()}\n\n{bert_tgt_preds}\n{word_to_tags}\n\n{word_pair_dict}")

        # Now map the Arguments ...
        gold_tgt_args, bert_tgt_args = [], []
        for predicate, arguments in bert_tgt_pred_arg_struct.items():
            for arg in arguments:
                arg_ix, arg_head, arg_tag = arg
                # if DE_head_corr_dict[arg_ix]: # EXPERIMENTAL!
                bert_tgt_args.append(f"{arg_tag}_{arg_head}")

        gold_tgt_preds, gold_tgt_pred_arg_struct = [], {}
        for predicate, arguments in t.argument_structure.items():
            ix, w, sense, _ = predicate
            gold_tgt_pred_arg_struct[(ix, w, sense)] = []
            gold_tgt_preds.append(f"{ix}_{w}_{sense}")
            for arg in arguments:
                arg_ix, _, arg_tag, arg_head = arg.get()
                gold_tgt_pred_arg_struct[(ix, w, sense)].append((arg_ix, arg_head, arg_tag))
                gold_tgt_args.append(f"{arg_tag}_{arg_head}")

        # *** Populate the CoNLL Template with the Obtained information ***
        t.tokens = populate_conll_template(bert_tgt_pred_arg_struct, word_to_tags, token_templates=t.tokens)
        new_annos.append(t)

        # *** Compare the Obtained information with the Gold Data if given!***
        if compare_gold: 
            em_prd, em_arg, em_all = evaluate_projections(bert_tgt_preds, gold_tgt_preds, bert_tgt_args, gold_tgt_args)
            _add_to_eval_dicts(em_prd, pred_excess, pred_missed, pred_match)
            _add_to_eval_dicts(em_arg, arg_excess, arg_missed, arg_match)
            _add_to_eval_dicts(em_all, all_excess, all_missed, all_match)

        # -- Follow which Predicates were identified per sentence! --
        all_bert_preds.append(bert_tgt_preds)
    
    if compare_gold:
        # Overall Metrics
        print("\n\n---------------  PREDICATES --------------------\n")
        ue.print_overall_metrics(pred_excess, pred_missed, pred_match)
        print("\n\n---------------  ARGUMENTS --------------------\n")
        ue.print_overall_metrics(arg_excess, arg_missed, arg_match)
        print("\n\n---------------  PRED-ARG ALL --------------------\n")
        ue.print_overall_metrics(all_excess, all_missed, all_match)

    return new_annos


def _add_to_eval_dicts(eval_metrics, arg_excess, arg_missed, arg_match):
    for arg in eval_metrics["excess"]:
        arg_excess[arg] += 1
    for arg in eval_metrics["missed"]:
        arg_missed[arg] += 1
    for arg in eval_metrics["match"]:
        arg_match[arg] += 1


def evaluate_projections(bert_tgt_preds, gold_tgt_preds, bert_tgt_args, gold_tgt_args):
    # Evaluate Predicate Projections
    eval_metrics_preds = ue.evaluate_tagset(set(["V" + "_" + g.split("_")[0] for g in gold_tgt_preds]),
                                        set(["V" + "_" + b.split("_")[0] for b in bert_tgt_preds]),
                                        (-1, "<NONE>"),
                                        (-1, "<NONE>"), False, False)
    # Evaluate Argument Projections
    eval_metrics_args = ue.evaluate_tagset(set(gold_tgt_args), set(bert_tgt_args), (-1, "<NONE>"),
                                        (-1, "<NONE>"), False, False)
    # Evaluate ALL Projections Together (Preds + Args)
    eval_metrics_all = {"excess": [], "missed": [], "match": []}
    for key, val in eval_metrics_preds.items():
        eval_metrics_all[key] += val
    for key, val in eval_metrics_args.items():
        eval_metrics_all[key] += val

    logging.info("GOLD PREDS:" + str(gold_tgt_preds))
    logging.info("BERT PREDS:" + str(bert_tgt_preds))
    logging.info("GOLD ARGS:" + str(gold_tgt_args))
    logging.info("BERT ARGS:" + str(bert_tgt_args))
    return eval_metrics_preds, eval_metrics_args, eval_metrics_all


if __name__ == "__main__":
    """
    So we have SRC_annotated and TGT_aconll
    We score and then we [OPTIONALLY] compare vs ground-truth.
    
    RUN EXAMPLE: 
        python project_srl_annotations.py -s trial_data/mini_X-SRL_Test_EN.conll \
        -t trial_data/mini_X-SRL_Test_ES.conll --tgt_lang ES --align_mode S2T --tgt_has_gold True
  
    """

    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_file', help='Filepath containing the Source Side CoNLL Annotations', required=True)
    parser.add_argument('-t', '--tgt_file', help='Filepath containing the Target Side CoNLL Annotations', required=True)
    parser.add_argument('-l', '--tgt_lang', required=True)
    parser.add_argument('-k', '--k_candidates', default=2, help="The top-most similar word candidates to consider in alignments")
    parser.add_argument('-a', '--align_mode', required=True) # BERT-S2T | BERT-INTER | BERT-T2S | INTERSECT | S2T
    parser.add_argument('-g', '--tgt_has_gold', default="False", help="For Evaluation purposes! Only make true when the TGT CoNLL also has Annotations")
    parser.add_argument('-f', '--indices_filter', default="False", help="Use this flaf to pair the original EN with the TGT valid ids inside the .id_ref file")
    args = parser.parse_args()

    console_hdlr = logging.StreamHandler(sys.stdout)
    file_hdlr = logging.FileHandler(filename=f"logs/Sentence_Scorers_{args.tgt_lang}_{args.align_mode}.log")
    logging.basicConfig(level=logging.INFO, handlers=[console_hdlr, file_hdlr])
    logging.info("Start Logging")


    src_annos = read_conll(args.src_file, conll_token=CoNLLUP_Token, splitter="\t") # <EN> Source
    tgt_annos = read_conll(args.tgt_file, conll_token=CoNLLUP_Token, splitter="\t") # <DE/ES/FR> Target
    
    # This step is necessary if the SRC is English to match the valid translation ID's of the Target
    if get_bool_value(args.indices_filter):
        print("Pairing SRC-TGT sentences using the .id_ref file...")
        tgt_indices, qualities = get_valid_indices(f"{args.tgt_file}.id_ref")
        src_annos = filter_annos(src_annos, tgt_indices)
    else:
        tgt_indices, qualities = range(len(tgt_annos)), None

    TGT_LANG = args.tgt_lang
    SRC_SPACY = get_spacy_lang(lang="EN", use_large_model=False)
    TGT_SPACY = get_spacy_lang(lang=TGT_LANG, use_large_model=False)
    TGT_HAS_GOLD = get_bool_value(args.tgt_has_gold)

    assert len(src_annos) == len(tgt_annos), f"There are {len(src_annos)} sources and {len(tgt_annos)} target sentences. Maybe you forgot to activate the --indices_filter flag?"

    new_tgt_annos = transfer_annotations(src_annos, 
                                        tgt_annos, 
                                        args.tgt_lang, 
                                        alignment_mode=args.align_mode, 
                                        k=args.k_candidates,
                                        compare_gold=TGT_HAS_GOLD)
    write_conll_annos(new_tgt_annos, outname=f"{args.tgt_file}.{args.align_mode}.populated")
