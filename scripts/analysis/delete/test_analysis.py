import argparse
import numpy as np
import sys, subprocess
from scipy import stats

def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate text edit distance.')
    parser.add_argument('--result-path', dest='result_path', type=str, required=True, help=('Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'))
    parser.add_argument('--output-path', dest="out_path", type=str, default='top_confusions.csv', help=('Output file path, default=top_confusions.csv'))
    parameters = parser.parse_args(args)
    return parameters

def open_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

def main(args):
    params = process_args(args)
    filename = params.result_path
    testlines = open_file(filename)
    src = open_file('../data/im2text/src-test.txt')
    results = []
    for id, line in enumerate(testlines):
        items = line.split('\t')
        if len(items)==5:
            gold = items[1]
            pred = items[2]
            if pred.strip()=='':
                results.append((0.0, len(gold.split())))
                continue
            if items[0] in src:
                tmpgold = open('.tmp.goldsingle.txt','w')
                tmppred = open('.tmp.predsingle.txt','w')
                tmpgold.write(gold+'\n')
                tmpgold.close()
                tmppred.write(pred+'\n')
                tmppred.close()
                #proc = subprocess.Popen('perl third_party/multi-bleu.perl %s < %s'%('.tmp.goldsingle.txt','.tmp.predsingle.txt'), shell=True, stdout=subprocess.PIPE)
                #bleu = proc.stdout.read()
                bleu = float(str(subprocess.check_output('perl third_party/multi-bleu.perl %s < %s'%('.tmp.goldsingle.txt','.tmp.predsingle.txt'), shell=True)).split()[2].replace(',',''))
                results.append((bleu,len(gold.split())))
                #results.append(len(gold.split()))

    results = np.asarray(results)
    bins = (0,30,40,50,60,75,100,150,200)
    binned = np.digitize(results, bins)
    results = np.hstack((results, binned))
    results = np.delete(results, 2, 1)
    np.savetxt(params.out_path, results, delimiter=",")

if __name__ == '__main__':
    main(sys.argv[1:])

