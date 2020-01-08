import numpy
import sys
import subprocess as sp

def open_file(filename):
    with open(filename) as f:
            return [line.strip() for line in f]

def write_line(filename, line):
    with open(filename, 'w') as f:
        f.write(line)

def write_lines(filename, data):
        with open(filename, 'w') as f:
            c = 0
            for entry in data:
                f.write(entry)
                f.write('\n')
                c+=1
            print('Wrote '+str(c)+' lines to '+ filename)

## Script to process data and arrange it according to OCROPUS requirements

assert len(sys.argv)==4, 'Must use 3 arguments. Syntax: data_proc_ocropus.py input_filename labels_filename target_directory'

filenames = sys.argv[1]
lbl_filename = sys.argv[2]
target_directory = sys.argv[3]

lines = open_file(filenames)
labels = open_file(lbl_filename)

names = []

idx=0
for idx, line in enumerate(lines):
    name = '../data/im2text/images/' + line
    names.append('data/train/'+line)
    command = 'cp ' + name + ' ' + target_directory + line
    sp.run(command, shell='True', check='True')
    label_fname = target_directory + line.replace('.png','.gt.txt')
    write_line(label_fname, labels[idx])
print('Read '+str(idx)+' lines from ' + filenames)
write_lines(target_directory+'filenames.txt', names)
