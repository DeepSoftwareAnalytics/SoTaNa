- Google bleu without smooth ---- nltk without smooth (ignoring about 1e-6 numerical precision difference)
- Google bleu with smooth ---- nltk with SmoothingFunctions.method2
  
  Actually there is a little difference (about 0.2) in these two algorithms, which is when in nltk, the function returns 0 if 1-gram overlap is 0, but google_bleu still use smoothing function.

  I think nltk is better.
- Google bleu is just the other smooth bleu in CoCoGum;
- Run `python test_scripts/cmp_google_nltk.py` to see the comparison of these to algorithms;
  - Uncomment the added part in [nltk_bleu.py](acl20_eval/bleu/google_bleu.py) and comment the original part, you can see the results are almost the same if ignoring some numerical precision difference;
  
  
Running result:
```
$ python run_comparison.py
        Codebert bleu: 44.35852901
        Codenn bleu: 44.35852901
        Google bleu with smooth [corpus, avg]: 0.38746092 0.44696539
        NLTK bleu with smooth method2 [corpus, avg]: 0.38745998 0.44463355
        NLTK bleu with smooth method4 [corpus, avg]: 0.38745599 0.51972231
/anaconda/envs/py37_default/lib/python3.7/site-packages/nltk/translate/bleu_score.py:516: UserWarning:
The hypothesis contains 0 counts of 2-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()
  warnings.warn(_msg)
/anaconda/envs/py37_default/lib/python3.7/site-packages/nltk/translate/bleu_score.py:516: UserWarning:
The hypothesis contains 0 counts of 3-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()
  warnings.warn(_msg)
/anaconda/envs/py37_default/lib/python3.7/site-packages/nltk/translate/bleu_score.py:516: UserWarning:
The hypothesis contains 0 counts of 4-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()
  warnings.warn(_msg)
        NLTK bleu without smooth [corpus, avg]: 0.38745599 0.37934548
```
