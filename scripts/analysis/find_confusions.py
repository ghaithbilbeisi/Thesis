import os, sys, argparse, logging
import difflib
import numpy as np

def process_args(args):
    parser = argparse.ArgumentParser(description='Get List of Symbol Confusions.')

    parser.add_argument('--result-path', dest='result_path',
                        type=str, required=True,
                        help=('Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'
                        ))

    parser.add_argument('--output-path', dest="out_path",
                        type=str, default='top_confusions.csv',
                        help=('Output file path, default=top_confusions.csv' 
                        ))
    parameters = parser.parse_args(args)
    return parameters

def add_confusions(word1, word2, confusion_array):
    if len(word1) != 1:
        if len(word2) != 1:
            for w1 in word1:
                for w2 in word2:
                    confusion_array.append((w1, '@', w2))
        else:
            for w in word1:
                confusion_array.append((w, '@', word2[0]))
    elif len(word2) != 1:
        for w in word2:
            confusion_array.append((word1[0], '@', w))
    else:
        confusion_array.append((word1[0], '@', word2[0]))


## Script to find the symbol confusions for a model using some Edit Distance functionality

def main(args):
    parameters = process_args(args)

    result_file = parameters.result_path
    total_ref = 0
    total_edit_distance = 0
    all_confusions = []
    with open(result_file) as fin:
        for idx,line in enumerate(fin):
            #if idx % 100 == 0:
                #print (idx)
            items = line.strip().split('\t')
            if len(items) == 5 and idx < 10:
                img_path, label_gold, label_pred, score_pred, score_gold = items
                l_pred = label_pred.strip()
                l_gold = label_gold.strip()
                tokens_pred = l_pred.split(' ')
                tokens_gold = l_gold.split(' ')
                #
                confusions = []
                matcher = difflib.SequenceMatcher(None, tokens_gold, tokens_pred)
                mb = matcher.get_matching_blocks()
                print(l_gold)
                print(l_pred)
                print(mb)
                for idm, m in enumerate(mb[:-1]):
                    if idm == 0 and (m[0] != 0 or m[1] != 0):
                        confusions.append((tokens_gold[0 : m[0]], tokens_pred[0 : m[1]]))

                    if m[0]+m[2] == len(tokens_gold):
                        continue

                    confusions.append((tokens_gold[m[0]+m[2] : mb[idm+1][0]], tokens_pred[m[1]+m[2] : mb[idm+1][1]]))
                c = 0
                while c < len(confusions):
                    if not confusions[c][0] and c < len(confusions)-1 and not confusions[c+1][1] and confusions[c+1][0]:
                        #all_confusions.append((confusions[c+1][0], '@', confusions[c][1], '@', len(confusions[c+1][0])))
                        add_confusions(confusions[c+1][0], confusions[c][1], all_confusions)
                        c += 2
                        continue
                    if not confusions[c][1] and c < len(confusions)-1 and not confusions[c+1][0] and confusions[c+1][1]:
                        #all_confusions.append((confusions[c][0], '@', confusions[c+1][1], '@', len(confusions[c][0])))
                        add_confusions(confusions[c][0], confusions[c+1][1], all_confusions)
                        c += 2
                        continue
                    #all_confusions.append((confusions[c][0],'@',confusions[c][1], '@', len(confusions[c][0])))
                    add_confusions(confusions[c][0], confusions[c][1], all_confusions)
                    c += 1
    #print(all_confusions)
    #exit()
    all_confusions = np.asarray(all_confusions)
    np.savetxt(parameters.out_path, all_confusions, fmt='%s')
    
   
if __name__ == '__main__':
    main(sys.argv[1:])
