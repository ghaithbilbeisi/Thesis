import argparse
import numpy as np
import sys, subprocess
#from scipy import stats

def process_args(args):
    parser = argparse.ArgumentParser(description='Produce BLEU and sequence length data for two models.')
    parser.add_argument('--result-path1', dest='result_path1', type=str, required=True, help=('Result of first file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'))
    parser.add_argument('--result-path2', dest='result_path2', type=str, required=True, help=('Result of second file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'))
    parser.add_argument('--output-path', dest="out_path", type=str, default='comparison.csv', help=('Output file path, default=top_confusions.csv'))
    parameters = parser.parse_args(args)
    return parameters

def open_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

def measure_bleu_single(gold, pred):
    tmpgold = open('.tmp.comp_goldsingle.txt','w')
    tmppred = open('.tmp.comp_predsingle.txt','w')
    tmpgold.write(gold+'\n')
    tmpgold.close()
    tmppred.write(pred+'\n')
    tmppred.close()
    bleu = float(str(subprocess.check_output('perl third_party/multi-bleu.perl %s < %s'%('.tmp.comp_goldsingle.txt','.tmp.comp_predsingle.txt'), shell=True)).split()[2].replace(',',''))
    return bleu

## Script that produces bleu results for each image using each of the two result files specified along with sequence length information.

def main(args):
    params = process_args(args)
    filename1 = params.result_path1
    filename2 = params.result_path2
    testlines1 = open_file(filename1)
    testlines2 = open_file(filename2)
    src = open_file('../data/im2text/src-test.txt')
    results = []
    for id, line in enumerate(testlines1):
        items1 = line.split('\t')
        items2 = testlines2[id].split('\t')
        if len(items1)==5 and len(items2)==5 and items1[0] in src:
            gold1 = items1[1]
            pred1 = items1[2]
            gold2 = items2[1]
            pred2 = items2[2]
            name = items1[0]
            if pred1.strip()=='':
                bleu1 = 0.0
                if pred2.strip()=='':
                    bleu2 = 0.0
                    #results.append((items1[0], 0.0, 0.0, 0.0, len(gold1.split())))
                    #continue
                else:
                    bleu2 = measure_bleu_single(gold2, pred2)
                    #results.append((items1[0], 0.0, bleu2, 0.0 - bleu2, len(gold1.split())))
                    #continue
            elif pred2.strip()=='':
                bleu1 = measure_bleu_single(gold1, pred1)
                bleu2 = 0.0
            else:
                bleu1 = measure_bleu_single(gold1, pred1)
                bleu2 = measure_bleu_single(gold2, pred2)
            results.append((name, bleu1, bleu2, bleu1 - bleu2, len(gold1.split())))
            #results.append(len(gold.split()))

    results = np.asarray(results, dtype=object)
    bins = np.array([0,30,40,50,60,75,100,150,200,250,300])
    binned = np.digitize(results[:,4], bins)
    results = np.hstack((results, np.reshape(binned,(-1,1))))
    np.savetxt(params.out_path, results, delimiter=",", fmt='%s')

if __name__ == '__main__':
    main(sys.argv[1:])

