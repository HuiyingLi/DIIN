"""
This file generates tokenization, POS features, stemmed words from a spacy service.
"""
import util
import json
import pdb
import glob
from util.spacy.spacy_client import SpacyClient
import nltk
from nltk.corpus import wordnet as wn
client = SpacyClient()
req_batch_size = 200

PADDING = "<PAD>"
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_Tagging_spacy = [PADDING, '-LRB-','-RRB-', ',', ':', '.', "''", '""', '``', '#', '$', 'ADD', 'AFX','BES','CC','CD','DT','EX','FW','GW','HVS','HYPH','IN','JJ','JJR','JJS','LS','MD','NFP','NIL','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','_SP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','XX']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}
POS_dict_spacy = {pos:i for i, pos in enumerate(POS_Tagging_spacy)}

def extract_pos(sentobj):
    pos_vec = []
    for sent in sentobj['sentences']:
        pos = [tkobj['tag'] for tkobj in sent['tokens']]
        pos = [POS_dict_spacy.get(tag, 0) for idx, tag in enumerate(pos)]
        pos_vec+=pos
    return pos_vec

def is_exact_lemma_match(lemma1, lemma2):
    lemma1 = lemma1.lower()
    lemma2 = lemma2.lower()
    if lemma1==lemma2:
        return True
    for synsets in wn.synsets(lemma2):
        for lemma in synsets.lemma_names():
            if lemma1 == lemma:
                return True
    return False

def extract_tokens(sentobj):
    tokenlst = []
    msg = sentobj['message']['text']
    for sent in sentobj['sentences']:
        for tkobj in sent['tokens']:
            idx = tkobj['idx'] if 'idx' in tkobj else 0
            tokenlst.append(msg[idx:(idx + tkobj['len'])])
    return tokenlst

def extract_lemma(sentobj):
    lemmalst=[]
    for sent in sentobj['sentences']:
        lemmalst+=[tkobj['lemma'] for tkobj in sent['tokens']]
    return lemmalst

def nli_jsonl_reader(fpath):
    for l in open(fpath, 'r', encoding='utf8'):
        o = json.loads(l.strip())
        yield o['sentence1'], o['sentence2'], {k:o[k] for k in o if ('sentence' not in k and 'annotator' not in k)}

def annotate_corpus(fplst):
    """
    The input is a list of files of paraphrase pairs.
    The result will be written to file named with same prefix and different suffix.
    :return:
    """
    for fp in fplst:
        outputfp = fp[:fp.rfind('.')]+'.spacy.jsonl'
        req_batch=[]
        with open(outputfp, 'w', encoding='utf8') as outputf:
            for s1, s2, otherdict in nli_jsonl_reader(fp):
                req_batch.append(s1)
                req_batch.append(s2)
                if len(req_batch) == req_batch_size:
                    resp = client.annotate_input(req_batch)
                    assert len(resp)==req_batch_size
                    for i in range(len(resp)//2):
                        s1ann = json.loads(resp[i*2])
                        s2ann = json.loads(resp[i*2+1])
                        s1token = extract_tokens(s1ann)
                        s2token = extract_tokens(s2ann)
                        s1pos = extract_pos(s1ann)
                        s2pos = extract_pos(s2ann)
                        s1lemma = extract_lemma(s1ann)
                        s2lemma = extract_lemma(s2ann)
                        assert len(s1token) == len(s1pos) == len(s1lemma)
                        assert len(s2token) == len(s2pos) == len(s2lemma)
                        s1em = [0]*len(s1token)
                        s2em = [0]*len(s2token)
                        for i, word in enumerate(s1lemma):
                            for j, w2 in enumerate(s2lemma):
                                matched = is_exact_lemma_match(word, w2)
                                if matched:
                                    s1em[i] = 1
                                    s2em[j] = 1
                        o = {'sentence1':s1ann, 'sentence1_tokens': s1token, 'sentence1_pos': s1pos, 'sentence1_exact_match':s1em,
                             'sentence2':s2ann, 'sentence2_tokens': s2token, 'sentence2_pos': s2pos, 'sentence2_exact_match':s2em}
                        o = {**o, **otherdict}
                        outputf.writelines(json.dumps(o)+'\n')
                    req_batch=[]
        print("Done annotating file", fp)

if __name__ == '__main__':
    #annotate_corpus(glob.glob('/Users/huiyingli/Documents/DIIN/data/multinli_0.9/*.jsonl'))
    annotate_corpus(glob.glob('/Users/huiyingli/Documents/DIIN/data/snli_1.0/*.jsonl'))