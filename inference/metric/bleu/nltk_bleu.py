# modified from https://github.com/wasiahmad/NeuralCodeSum/blob/b2652e20cc8df6135b1b115b604752c7cbc94d56/c2nl/eval/bleu/nltk_bleu.py

import nltk
from nltk.translate.bleu_score import SmoothingFunction


# The funcom Ba score (https://github.com/mcmillco/funcom/blob/41c737903/bleu.py#%23L17) and 
# The 'Cumulative 4-gram BLEU (BLEU-4) c_bleu4 reported in CoCoGUM log equals to 
# nltk_corpus_bleu()[0]: NLTK bleu without smooth Corpus. 
# In brief: Ba = nltk.translate.bleu_score.corpus_bleu(refs, preds)
def nltk_corpus_bleu(hypotheses, references, order=4):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = nltk.translate.bleu(ref, hyp)
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hyps)
    return corpus_bleu, avg_score, ind_score


def nltk_corpus_bleu_smooth2(hypotheses, references, order=4):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = nltk.translate.bleu(ref, hyp, smoothing_function=cc.method2)
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hyps, smoothing_function=cc.method2)
    return corpus_bleu, avg_score, ind_score



#-------below is original https://github.com/wasiahmad/NeuralCodeSum/blob/master/c2nl/eval/bleu/nltk_bleu.py##
def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


# BLEU-EMSE(DeepCom) is nltk_corpus_bleu_smooth4 Avg.
def nltk_corpus_bleu_smooth4(hypotheses, references, order=4):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = nltk.translate.bleu(ref, hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hyps)
    return corpus_bleu, avg_score, ind_score


def nltk_bleus(refs, preds):
    c_bleu1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    c_bleu2 = corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0))
    c_bleu3 = corpus_bleu(refs, preds, weights=(1/3, 1/3, 1/3, 0))
    c_bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))
    i_bleu2 = corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    i_bleu3 = corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    i_bleu4 = corpus_bleu(refs, preds, weights=(0, 0, 0, 1))
#     ret  = 'Cumulative 1/2/3/4-gram BLEU: {}, {}, {}, {}\n'.format(c_bleu1, c_bleu2, c_bleu3, c_bleu4)
#     ret += 'Individual 1/2/3/4-gram BLEU: {}, {}, {}, {}\n'.format(c_bleu1, i_bleu2, i_bleu3, i_bleu4)
    return c_bleu1, c_bleu2, c_bleu3, c_bleu4, i_bleu2, i_bleu3, i_bleu4