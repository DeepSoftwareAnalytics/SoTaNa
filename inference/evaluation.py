import json
import numpy as np
import sys
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

sys.path.append("metric")
from metric.codenn_bleu import codenn_smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from metric.cider.cider import Cider
from bleu.google_bleu import compute_bleu
from bleu.rencos_bleu import Bleu as recos_bleu
import warnings
import argparse
import logging
# import prettytable as pt

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def bleus(refs, preds):

    # emse_bleu
    all_score = 0.0
    count = 0
    for r, p in zip(refs, preds):
        score = nltk.translate.bleu(r, p, smoothing_function=SmoothingFunction().method4)
        all_score += score
        count += 1
        emse_bleu = round(all_score / count * 100, 2)


    bleus_dict = {
                  'BLEU-DC': emse_bleu,
                
    }
    return bleus_dict

def Sentence_BLUE_SM2(refs, preds):

    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
    except:
        bleu_list = [0, 0, 0, 0]
    codenn_bleu = bleu_list[0]

    Blue = round(codenn_bleu, 4)

    return Blue


def read_to_list(filename):
    f = open(filename, 'r',encoding="utf-8")
    res = []
    for row in f:
        # (rid, text) = row.split('\t')
        if len(row) > 0:
            res.append(row.lower().split())
    print(len(res))
    return res

def metetor_rouge_cider(refs, preds):

    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        try:
            refs_dict[i] = [" ".join(refs[i][0])]
        except IndexError:
            print(i)
            print(refs[i])
            exit(11)

        
    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    # print("Meteor: ", round(score_Meteor*100,2))

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    # print("Rouge-L: ", round(score_Rouge*100,2))

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    # print("Cider: ",round(score_Cider,2) )

    results = {"Meteor": round(score_Meteor*100,2),
    "Rouge-L": round(score_Rouge*100,2),
    "Cider": round(score_Cider,2)
    }
    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs_filename', type=str, default="test/test.gold", required=False)
    parser.add_argument('--preds_filename', type=str, default="test/test.pred", required=False)
    args = parser.parse_args()
    refs = read_to_list(args.refs_filename)
    refs = [[t[1:]] for t in refs]
    preds = read_to_list(args.preds_filename)
    preds = [t[1:] for t in preds]
    print("A ref is: %s"% (" ".join(refs[0][0])) )
    print("A preds is: %s"% (" ".join(preds[0])) )
    score = bleus(refs, preds)
    print(score)
    metetor_rouge_cider(refs, preds)


if __name__ == '__main__':
    main()