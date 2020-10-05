from tabulate import tabulate # https://pypi.org/project/tabulate/

def print_overall_metrics(arg_excess, arg_missed, arg_match):
    for x in arg_match.items():
        print(x)
    processed_args = set()
    results = []
    tot_excess, tot_missed, tot_match = 0, 0, 0
    for arg, count in arg_match.items():
        excess = arg_excess.get(arg, 0)
        missed = arg_missed.get(arg, 0)
        p,r,f = get_metrics(false_pos=excess, false_neg=missed, true_pos=count)
        processed_args.add(arg)
        results.append((arg, count, excess, missed, p, r, f))
        tot_excess += excess
        tot_missed += missed
        tot_match += count
    for arg, count in arg_excess.items():
        if arg not in processed_args:
            excess = count
            missed = arg_missed.get(arg, 0)
            correct = arg_match.get(arg, 0)
            p, r, f = get_metrics(false_pos=excess, false_neg=missed, true_pos=correct) # p,r,f = 0,0,0
            processed_args.add(arg)
            results.append((arg, correct, excess, missed, p, r, f))
            tot_excess += excess
            tot_missed += missed
            tot_match += correct
    for arg, count in arg_missed.items():
        if arg not in processed_args:
            excess = arg_excess.get(arg, 0)
            correct = arg_match.get(arg, 0)
            missed = count
            p, r, f = get_metrics(false_pos=excess, false_neg=missed, true_pos=correct) # p,r,f = 0,0,0
            results.append((arg, correct, excess, missed, p, r, f))
            tot_excess += excess
            tot_missed += missed
            tot_match += correct
    results = sorted(results, key= lambda x: x[0])

    prec, rec, F1 = get_metrics(false_pos=tot_excess, false_neg=tot_missed, true_pos=tot_match)

    print("\n--- OVERALL ---\nCorrect: {0}\tExcess: {1}\tMissed: {2}\nPrecision: {3:.2f}\t\tRecall: {4:.2f}\nF1: {5:.2f}\n".format(tot_match, tot_excess, tot_missed, prec, rec, F1))
    print(tabulate(results, headers=["corr.", "excess", "missed", "prec.", "rec.", "F1"], floatfmt=".2f"))


def get_metrics(false_pos, false_neg, true_pos):
    _denom = true_pos + false_pos
    precision = true_pos / _denom if _denom else 0
    _denom = true_pos + false_neg
    recall = true_pos / _denom if _denom else 0
    _denom = precision + recall
    F1 = 2 * ((precision * recall) / _denom) if _denom else 0
    return precision*100, recall*100, F1*100


def evaluate_tagset(gld, sys, gld_pred, sys_pred, consider_verb_token, consider_verb_position):
    gld_ix, gld_pred = gld_pred
    sys_ix, sys_pred = sys_pred
    if consider_verb_token:
        if gld_pred == "<NONE>" and "V" in sys:  # Count all tags as EXCESS
            print("MisMatch!", gld_pred, sys_pred)
            return {"excess": [x.split("_")[0] for x in sys], "missed": [], "match": []}
        elif gld_pred.lower() != sys_pred.lower():  # Predicate Missmatch: Count all tags as MISSING, Including Predicate!
            return {"excess": [], "missed": [x.split("_")[0] for x in sys] + ["V"], "match": []}

    if consider_verb_position and gld_ix != sys_ix: # Predicate Missmatch: Count all tags as MISSING, Including Predicate!
        return {"excess": [], "missed": [x.split("_")[0] for x in sys] + ["V"], "match": []}

    # print("SYS", sys)
    # print("GLD", gld)
    excess = sys - gld  # False Positives
    missed = gld - sys  # False Negatives
    true_pos = sys.intersection(gld)
    # print("Excess",excess)
    # print("Missed",missed)
    # print("TruePos",true_pos)
    # print("---------------")
    eval_obj = {"excess": [x.split("_")[0] for x in excess],
                "missed": [x.split("_")[0] for x in missed],
                "match": [x.split("_")[0] for x in true_pos]}
    return eval_obj