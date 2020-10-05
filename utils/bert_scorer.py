from bert_score import score
from bert_score.utils import get_bert_embedding, sent_encode
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import numpy as np
import torch


def get_bert_model(model_type, basic_tokenize):
    tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=False, do_basic_tokenize=basic_tokenize)
    model = BertModel.from_pretrained(model_type)
    model.eval()
    return model, tokenizer


def bert_similarities(source, target, model, tokenizer):
    matrix = get_confusion_matrix(candidate=target, reference=source, model=model, tokenizer=tokenizer)
    return matrix


def get_confusion_matrix(candidate, reference, model, tokenizer):
    """
    Get Confusion Matrix containing Cosine Similarity of Words.
    
    :param: candidate (str): a candidate sentence
    :param: reference (str): a reference sentence
    :param: model (BertModel): bert specification (in our case we always use mBERT for cross-lingual alignment)
    :param: tokenizer (BertTokenizer): To break sentences into BERT WordPieces
    """
    assert isinstance(candidate, str)
    assert isinstance(reference, str)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    idf_dict = defaultdict(lambda: 1.)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    hyp_embedding, _, _ = get_bert_embedding([candidate], model, tokenizer, idf_dict,
                                                         device=device, all_layers=False)
    ref_embedding, _, _ = get_bert_embedding([reference], model, tokenizer, idf_dict,
                                                         device=device, all_layers=False)
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim.squeeze(0).cpu()

    # remove [CLS] and [SEP] tokens
    r_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, reference)][1:-1]
    h_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, candidate)][1:-1]
    sim = sim[1:-1,1:-1]

    return r_tokens, h_tokens, sim


def get_full_word_mapping(wordpieces, full_words):
    # Mappping goes from FullWord_Index to WordPiece Index
    mapping_ix, mapping = {}, {}
    # The inverse will be for each WordPiece Index to the Original Index in the FullWord Sentence
    inverse_mapping_ix = {}
    # print(full_words)
    # print(wordpieces)
    for ix, wp in enumerate(wordpieces):
        if "##" in wp:
            mapping[len(mapping) - 1].append(wp)
            mapping_ix[len(mapping) - 1].append(ix)
        else:
            mapping[len(mapping)] = [wp]
            mapping_ix[len(mapping_ix)] = [ix]

    for word_ix, wp_ix_lst in mapping_ix.items():
        for wp_ix in wp_ix_lst:
            inverse_mapping_ix[wp_ix] = word_ix
    # print("wp2full", mapping)
    return mapping_ix, inverse_mapping_ix


def get_most_similar_pairs(src_anno, tgt_anno, src_bert_tokens, tgt_bert_tokens, 
                            sim_matrix, get_best, alignment_mode, verbose=False):
    """
    :param src_anno (AnnotatedSentence): CoNLL Annotated Source Sentence Object
    :param tgt_anno (AnnotatedSentence): CoNLL Annotated Target Sentence Object
    :param src_bert_tokens (List): Source-Language BERT WordPieces
    :param tgt_bert_tokens (List): Target-Language BERT WordPieces
    :param sim_matrix (2D Tensor): BERTScore Similarity Matrix (rows=tgt_bert_tokens, cols=src_bert_tokens)
    :param alignment_mode (str): String to indicate which alignment to return. Options: INTERSECT | S2T
    :param verbose (bool): To print details on the console
    :return:
    """

    def _get_best_tgt_alignment(alignments, current_mapping):
        if len(alignments) == 1: return alignments[0], current_mapping
        inv_mapping = defaultdict(list)
        for src_word, (tgt_word, score) in current_mapping.items():
            inv_mapping[tgt_word].append((src_word, score))
        align_by_score = sorted(alignments, key=lambda x: x["score"], reverse=True) # From Best to Worst
        best_alignment = align_by_score[0] if align_by_score[0]["word_full"] not in inv_mapping else align_by_score[1]
        for al in align_by_score:
            if al["word_full"] == "[UNK]": continue
            original_pos = src_pos[al["src_full_ix"]]
            w_ix, _ = al["word_full"].split("_")
            if original_pos.startswith("V") and tgt_pos[int(w_ix)].startswith("V"):
                al["score"] = al["score"] + 1
            if al["score"] >= best_alignment["score"]:
                if al["word_full"] in inv_mapping:
                    for curr_src, curr_score in inv_mapping[al["word_full"]]:
                        if al["score"] >= curr_score:
                            best_alignment = al
                            del current_mapping[curr_src]
                        else:
                            best_alignment = {"word_full": current_mapping[curr_src], "score": curr_score}
                else:
                    best_alignment = al

        return best_alignment, current_mapping

    def _get_full_src_word(wp_index):
        full_ix = src_inv_mapping.get(wp_index, -1)
        if full_ix >= 0:
            return f"{full_ix}_{src_full_tokens[full_ix]}"
        else:
            return "[UNK]"

    def _get_full_tgt_word(wp_index):
        full_ix = tgt_inv_mapping.get(wp_index, -1)
        if full_ix >= 0:
            return f"{full_ix}_{tgt_full_tokens[full_ix]}"
        else:
            return "[UNK]"

    src_full_tokens = src_anno.get_tokens() # gets only the words of AnnotatedSentence
    tgt_full_tokens = tgt_anno.get_tokens()
    src_pos = [tok.pos_tag for tok in src_anno.tokens]
    tgt_pos = [tok.pos_tag for tok in tgt_anno.tokens] 

    # Keep track of the BEST Aligned SRC<->TGT Pairs
    s2t_best, t2s_best = defaultdict(list), defaultdict(list)
    # Get All SRC-TGT Mappings
    _, src_inv_mapping = get_full_word_mapping(src_bert_tokens, src_full_tokens) # _ = src_mapping
    _, tgt_inv_mapping = get_full_word_mapping(tgt_bert_tokens, tgt_full_tokens) # _ = tgt_mapping


    # Matrix SRC -> TGT. Get Alignments
    for i, row in enumerate(sim_matrix.transpose(1, 0)):
        top = np.argsort(-row)[:get_best]
        full_src_word_ix = src_inv_mapping.get(i, -1)
        alignments = [{"word_full": _get_full_tgt_word(j.item()),
                       "word_piece": tgt_bert_tokens[j.item()],
                       "src_full_ix": full_src_word_ix,
                       "tgt_full_ix": tgt_inv_mapping.get(j.item(), -1),
                       "score": sim_matrix[j][i].item()} for j in top]
        if full_src_word_ix >= 0: s2t_best[f"{full_src_word_ix}_{src_full_tokens[full_src_word_ix]}"] += alignments

    # Matrix TGT -> SRC. Get Alignments
    for i, row in enumerate(sim_matrix):
        top = np.argsort(-row)[:get_best]
        full_tgt_word_ix = tgt_inv_mapping.get(i, -1)
        alignments = [{"word_full": _get_full_src_word(j.item()),
                       "word_piece": src_bert_tokens[j.item()],
                       "src_full_ix": src_inv_mapping.get(j.item(), -1),
                       "tgt_full_ix": full_tgt_word_ix,
                       "score": sim_matrix[i][j].item()} for j in top]
        if full_tgt_word_ix >= 0: t2s_best[f"{full_tgt_word_ix}_{tgt_full_tokens[full_tgt_word_ix]}"] += alignments

    if verbose:
        print(src_bert_tokens)
        print(tgt_bert_tokens)
        print("\nS->T")
        [print(x) for x in s2t_best.items()]
        # print("\nT->S")
        # [print(x) for x in t2s_best.items()]

    # Get FINAL Alignments
    best_pairs = {}
    if alignment_mode == "INTER" or alignment_mode == "INTERSECT":
        print("Intersect Alignment")
        for s, t_objs in s2t_best.items():
            # t_obj = sorted(t_objs, key=lambda x: x["score"])[-1]
            t_obj, best_pairs = _get_best_tgt_alignment(t_objs, best_pairs)
            aligned_sources = t2s_best.get(t_obj["word_full"])
            if aligned_sources:
                aligned_s = sorted(aligned_sources, key=lambda x: x["score"])[-1]
                if aligned_s["word_full"] == s:
                    best_pairs[s] = (t_obj["word_full"], t_obj["score"])
    elif alignment_mode == "S2T":
        print("S2T Alignment")
        for s, t_objs in s2t_best.items():
            t_obj, best_pairs = _get_best_tgt_alignment(t_objs, best_pairs)
            if "_" in t_obj["word_full"]:
                best_pairs[s] = (t_obj["word_full"], t_obj["score"])
    elif alignment_mode == "BERT-INTER":
        print("Vanilla mBERT INTERSECT Alignment")
        for s, t_objs in s2t_best.items():
            t_obj = sorted(t_objs, key=lambda x: x["score"])[-1]
            aligned_sources = t2s_best.get(t_obj["word_full"])
            if aligned_sources:
                aligned_s = sorted(aligned_sources, key=lambda x: x["score"])[-1]
                if aligned_s["word_full"] == s:
                    best_pairs[s] = (t_obj["word_full"], t_obj["score"])
    elif alignment_mode == "BERT-S2T":
        print("Vanilla mBERT S2T Alignment")
        for s, t_objs in s2t_best.items():
            t_obj = sorted(t_objs, key=lambda x: x["score"])[-1]
            if "_" in t_obj["word_full"]:
                best_pairs[s] = (t_obj["word_full"], t_obj["score"])
    elif alignment_mode == "BERT-T2S":
        print("Vanilla mBERT T2S Alignment")
        for t, s_objs in t2s_best.items():
            s_obj = sorted(s_objs, key=lambda x: x["score"])[-1]
            if "_" in s_obj["word_full"]:
                best_pairs[s_obj["word_full"]] = (t, s_obj["score"])
    else:
        print(alignment_mode)
        raise NotImplementedError

    return best_pairs
