import numpy
import sys

def open_file(filename):
    with open(filename) as f:
            return [line.strip() for line in f]

def write_file(filename, data):
    with open(filename, 'w') as f:
        c = 0
        for entry in data:
            f.write(entry)
            f.write('\n')
            c+=1
        print('Wrote '+str(c)+' lines to '+ filename)

logfile = sys.argv[1]
srcfile = sys.argv[2]
outfile = sys.argv[3]
logs = open_file(logfile)
src = open_file(srcfile)
output = []

src_id = 0
lid = 0
while lid <= len(logs)-1:
    log = logs[lid]
    if log != '' and log.split()[0] == 'SENT':
        name = src[src_id]
        if len(logs[lid+1].split()) > 2:
            pred = logs[lid+1].split(' ',2)[2]
        else:
            pred = ''
        pred_score = logs[lid+2].split(' ',2)[2]
        if len(logs[lid+3].split()) > 2:
            gold = logs[lid+3].split(' ',2)[2]
        else:
            gold = ''
        gold_score = logs[lid+4].split(' ',2)[2]
        line = ''.join(name + '\t' + gold + '\t' + pred + '\t' + pred_score + '\t' + gold_score)
        #print(name + '\t' + gold + '\t' + pred + '\t' + pred_score + '\t' + gold_score)
        output.append(line)
        src_id += 1
        lid += 6
    else:
        lid += 1

write_file(outfile, output)