import argparse
import numpy as np
import sys, subprocess

def process_args(args):
    parser = argparse.ArgumentParser(description='Produce symbol frequency analysis data.')
    parser.add_argument('--result-path1', dest='result_path1', type=str, required=True, help=('Result of first file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'))
    parser.add_argument('--output-path', dest="out_path", type=str, default='freq_analysis.txt', help=('Output file path, default=freq_analysis.txt, save to txt then copy to csv or excel sheet since the \'=\' symbol will cause issues.'))
    parameters = parser.parse_args(args)
    return parameters

def open_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

def measure_bleu_single(gold, pred):
    tmpgold = open('.tmp.freq_goldsingle.txt','w')
    tmppred = open('.tmp.freq_predsingle.txt','w')
    tmpgold.write(gold+'\n')
    tmpgold.close()
    tmppred.write(pred+'\n')
    tmppred.close()
    bleu = float(str(subprocess.check_output('perl third_party/multi-bleu.perl %s < %s'%('.tmp.freq_goldsingle.txt','.tmp.freq_predsingle.txt'), shell=True)).split()[2].replace(',',''))
    return bleu

## A script to produce symbol frequency analysis data for a model

def main(args):
    params = process_args(args)
    filename1 = params.result_path1
    testlines = open_file(filename1)
    src = open_file('../data/im2text/src-test.txt')
    train = open_file('../data/im2text/tgt-train.txt')
    tmp = open_file('../data/im2text/vocab.txt')
    vocab = np.zeros((len(tmp), 4), dtype=object)
    vocab[:,0] = tmp
    c_all_tokens = 0 
    for line in train:
        for t in line.split(' '):
            c_all_tokens += 1
            for sid,symbol in enumerate(vocab):
                if t.strip() == symbol[0].strip():
                    symbol[1] += 1

    for id, line in enumerate(testlines):
        items = line.split('\t')
        if len(items)==5 and items[0] in src:
            gold = items[1]
            pred = items[2]
            name = items[0]
            if pred.strip()=='':
                bleu = 0.0
            else:
                bleu = measure_bleu_single(gold, pred)
            seq_counts = vocab[:, 3]
            for token in gold.split(' '):
                for idv, v in enumerate(vocab):
                    if token.strip()==v[0].strip():
                        #v[1] += 1
                        if seq_counts[idv] == v[3]:
                            v[2] += bleu
                            v[3] += 1
    vocab[:,1] = vocab[:,1]/c_all_tokens
    #vocab[:,2] = vocab[:,2]/vocab[:,3]
    np.savetxt(params.out_path, vocab, delimiter='*', fmt='%s')

if __name__ == '__main__':
    main(sys.argv[1:])

