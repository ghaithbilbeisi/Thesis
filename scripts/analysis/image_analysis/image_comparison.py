import sys, os, re, argparse, logging
import distance
import numpy as np
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../utils/'))
from utils import run
from image_utils import *
from pathlib import Path
from PIL import Image

TIMEOUT = 10

# replace \pmatrix with \begin{pmatrix}\end{pmatrix}
# replace \matrix with \begin{matrix}\end{matrix}
template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""


def process_args(args):
    parser = argparse.ArgumentParser(description='Render latex formulas for comparison. Note that we need to render both the predicted results, and the original formulas, since we need to make sure the same environment of rendering is used.')

    parser.add_argument('--result-path1', dest='result_path1',
                        type=str, required=True,
                        help=('Result file containing <img_path> <label_gold> <label_pred> <score_pred> <score_gold> per line. This should be set to the output file of the model.'
                        ))
    parser.add_argument('--result-path2', dest='result_path2',
                        type=str, required=True,
                        help=('Second result file.'
                        ))
    parser.add_argument('--output-path', dest='output_path',
                        type=str, required=True,
                        help=('Output file to store the results.'
                        ))
    parser.add_argument('--image-list', dest='image_list',
                        type=str, required=True,
                        help=('File containing the names of the images to be rendered.'
                        ))
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=14,
                        help=('Number of threads, default=4.'
                        ))
    parameters = parser.parse_args(args)
    return parameters

def open_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

def clean_sequence(l):
    l = l.strip()
    l = l.replace(r'\pmatrix', r'\mypmatrix')
    l = l.replace(r'\matrix', r'\mymatrix')
    # remove leading comments
    l = l.strip('%')
    if len(l) == 0:
        l = '\\hspace{1cm}'
    # \hspace {1 . 5 cm} -> \hspace {1.5cm}
    for space in ["hspace", "vspace"]:
        match = re.finditer(space + " {(.*?)}", l)
        if match:
            new_l = ""
            last = 0
            for m in match:
                new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                last = m.end(1)
            new_l = new_l + l[last:]
            l = new_l  
    return l  

def render_img(l, pre_name):
    tex_filename = pre_name+'.tex'
    log_filename = pre_name+'.log'
    aux_filename = pre_name+'.aux'
    with open(tex_filename, "w") as w: 
        w.write(template%l)
        #print(w, (template%l))
    run("pdflatex -interaction=nonstopmode %s  >/dev/null"%tex_filename, TIMEOUT)
    os.remove(tex_filename)
    if Path(log_filename).is_file():
        os.remove(log_filename)
    if Path(aux_filename).is_file():
        os.remove(aux_filename)
    pdf_filename = tex_filename[:-4]+'.pdf'
    png_filename = tex_filename[:-4]+'.png'
    if not os.path.exists(pdf_filename):
        print('cannot compile'+pre_name)
        return ""
    else:
        os.system("convert -density 200 -quality 100 %s %s"%(pdf_filename, png_filename))
        os.remove(pdf_filename)
        if os.path.exists(png_filename):
            return png_filename
            #crop_image(png_filename, output_path)
            #os.remove(png_filename)

# return (edit_distance, ref, match, match w/o)
def img_edit_distance(im1, im2):
    img_data1 = np.asarray(im1, dtype=np.uint8) # height, width
    img_data1 = np.transpose(img_data1)
    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]
    img_data1 = (img_data1<=128).astype(np.uint8)
    if im2:
        img_data2 = np.asarray(im2, dtype=np.uint8) # height, width
        img_data2 = np.transpose(img_data2)
        h2 = img_data2.shape[1]
        w2 = img_data2.shape[0]
        img_data2 = (img_data2<=128).astype(np.uint8)
    else:
        img_data2 = []
        h2 = h1
    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    elif h1 > h2:# pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2)) for item in img_data2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1)) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]

    seq1_int = [int(item,2) for item in seq1]
    seq2_int = [int(item,2) for item in seq2]
    
    edit_distance = distance.levenshtein(seq1_int, seq2_int)
    
    return (1 - (edit_distance/max(len(seq1_int),len(seq2_int))))

def main_parallel(line):
    img_name = line[0]
    g, l1, l2 = line[3:]
    pre_nameg = 'gold_'+img_name
    pre_name1 = '1_'+img_name
    pre_name2 = '2_'+img_name
    
    g = clean_sequence(g)
    l1 = clean_sequence(l1)
    l2 = clean_sequence(l2)

    img_nameg = render_img(g, pre_nameg)
    img_name1 = render_img(l1, pre_name1)
    img_name2 = render_img(l2, pre_name2)

    ed_acc1 = 0
    ed_acc2 = 0

    if img_nameg:
        img_g = Image.open(img_nameg).convert('L')
        if img_name1:
            img1 = Image.open(img_name1).convert('L')
            ed_acc1 = img_edit_distance(img_g, img1)
            #print(ed_acc1)
            os.remove(img_name1)
        if img_name2:
            img2 = Image.open(img_name2).convert('L')
            ed_acc2 = img_edit_distance(img_g, img2)
            #print(ed_acc2)
            os.remove(img_name2)
        os.remove(img_nameg)
    
    newline = (img_name, ed_acc1, ed_acc2, g, l1, l2)
    return newline

## Script that renders and evaluates a set of images for two models using the results files specified

def main(args):
    parameters = process_args(args)

    result_path1 = parameters.result_path1
    result_path2 = parameters.result_path2
    image_list = parameters.image_list

    results1 = open_file(result_path1)
    results2 = open_file(result_path2)
    img_list = open_file(image_list)

    lines = []
    c = 0
    for lid, line in enumerate(results1):
        items1 = line.strip().split('\t')
        if items1[0] in img_list:
            items2 = results2[lid].split('\t')
            lines.append((items1[0], 0, 0, items1[2], items1[1], items2[1]))
            lines[-1] = main_parallel(lines[-1])
            c += 1


    lines = np.asarray(lines)
    np.savetxt(parameters.output_path, lines[:,:3], delimiter="\t", fmt='%s')




if __name__ == '__main__':
    main(sys.argv[1:])
