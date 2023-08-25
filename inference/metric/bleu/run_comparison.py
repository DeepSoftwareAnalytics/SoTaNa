# Confirmed:
# (1) EMSE BLEU (DeepCom) (https://github.com/xing-hu/EMSE-DeepCom/blob/98bd6aac/scripts/evaluation.py) equals:
#     NLTK bleu with smooth method4 Avg.
# (2) The funcom Ba (also 'Cumulative 4-gram BLEU c_bleu4 reported in CoCoGUM log) equals: 
#     NLTK bleu without smooth Corpus.
# (3) In CoCoGUM, evaluate.CustomedBleu.bleu import _bleu equals:
#     google bleu with smooth Corpus.
# (4) Codebert bleu is adapted from Codenn bleu (with one line change: sciteamdrive2/yanlwang/bleu_test_all/codenn_bleu.pyL137), but the scores from the following experiments are the same.

import sys, json
from nltk_bleu import *
from google_bleu import corpus_bleu
from codebert_bleu import codebert_smooth_bleu
from codenn_bleu import codenn_smooth_bleu

references, predictions = {}, {}
with open('data/code2jdoc_test.json', 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        references[data['id']] = data['references']
        predictions[data['id']] = data['predictions']
        
preds, refs = [], []
for Id in predictions.keys():
    preds.append(predictions[Id][0])
    refs.append([references[Id][0]])
    

print('\tCodebert bleu: %.8f' % codebert_smooth_bleu(refs, preds)[0])
print('\tCodenn bleu: %.8f' % codenn_smooth_bleu(refs, preds)[0])
print('\tGoogle bleu with smooth [corpus, avg]: %.8f %.8f' % corpus_bleu(predictions, references)[0:2])
print('\tNLTK bleu with smooth method2 [corpus, avg]: %.8f %.8f' % nltk_corpus_bleu_smooth2(predictions, references)[0:2])
print('\tNLTK bleu with smooth method4 [corpus, avg]: %.8f %.8f' % nltk_corpus_bleu_smooth4(predictions, references)[0:2])
print('\tNLTK bleu without smooth [corpus, avg]: %.8f %.8f' % nltk_corpus_bleu(predictions, references)[0:2])
