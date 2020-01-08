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

## Script that transforms the original image list and label data files into the required format for ONMT to process and produce the data files

assert len(sys.argv)==5, 'Must use 4 arguments. Syntax: data_proc.py input_filename labels_filename sourcedata_filename targetdata_filename'

filename = sys.argv[1]
lbl_filename = sys.argv[2]
src_filename = sys.argv[3]
tgt_filename = sys.argv[4]

lines = open_file(filename)
labels = open_file(lbl_filename)
src = []
tgt = []

c = 0
for line in lines:
    l = line.split()
    label = labels[int(l[1])]
    if label.strip()!="":
        src.append(l[0])
        tgt.append(label)
        c+=1
print('Read '+str(c)+' lines from ' + filename)

write_file(src_filename, src)
write_file(tgt_filename, tgt)
