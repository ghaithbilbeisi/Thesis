import sys, os, re, argparse, logging
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../utils/'))
from utils import run
from image_utils import *
from multiprocessing.dummy import Pool as ThreadPool 
from pathlib import Path

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
    parser.add_argument('--output-dir1', dest='output_dir1',
                        type=str, required=True,
                        help=('Output directory to put the rendered images from the first results file.'
                        ))
    parser.add_argument('--output-dir2', dest='output_dir2',
                        type=str, required=True,
                        help=('Output directory to put the rendered images from the second results file.'
                        ))
    parser.add_argument('--image-list', dest='image_list',
                        type=str, required=True,
                        help=('File containing the names of the images to be rendered.'
                        ))
    parser.add_argument('--replace', dest='replace', action='store_true',
                        help=('Replace flag, if set to false, will ignore the already existing images.'
                        ))
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(replace=False)
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'
                        ))
    parameters = parser.parse_args(args)
    return parameters

## SCript that renders a set of images from the results files specified

def main(args):
    parameters = process_args(args)

    result_path1 = parameters.result_path1
    result_path2 = parameters.result_path2
    dir1 = parameters.output_dir1
    dir2 = parameters.output_dir2
    img_list_dir = parameters.image_list

    for dirname in [dir1, dir2]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    img_list = []
    with open(img_list_dir) as fin:
        for line in fin:
            img_list.append(line.strip())
    lines = []
    with open(result_path1) as fin:
        for line in fin:
            img_path, label_gold, label_pred, _, _ = line.strip().split('\t')
            if img_path in img_list and img_path == '761558c94d.png':
                print((img_path, label_pred, os.path.join(dir1, img_path), True))
                main_parallel((img_path, label_pred, os.path.join(dir1, img_path), True))
                exit()
                lines.append((img_path, label_pred, os.path.join(dir1, img_path), parameters.replace))
    with open(result_path2) as fin:
        for line in fin:
            img_path, label_gold, label_pred, _, _ = line.strip().split('\t')
            if img_path in img_list:
                lines.append((img_path, label_pred, os.path.join(dir2, img_path), parameters.replace))

    pool = ThreadPool(parameters.num_threads)
    results = pool.map(main_parallel, lines)
    pool.close() 
    pool.join() 

def output_err(output_path, i, reason, img):
    logging.info('ERROR: %s %s\n'%(img,reason))

def main_parallel(line):
    img_path, l, output_path, replace = line
    pre_name = output_path.replace('/', '_').replace('.','_')
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
    if replace or (not os.path.exists(output_path)):
        tex_filename = pre_name+'.tex'
        log_filename = pre_name+'.log'
        aux_filename = pre_name+'.aux'
        with open(tex_filename, "w") as w: 
            w.write(template%l)
            #print(w, (template%l))
        run("pdflatex -interaction=nonstopmode %s  >/dev/null"%tex_filename, TIMEOUT)
        exit()
        os.remove(tex_filename)
        if Path(log_filename).is_file():
            os.remove(log_filename)
        if Path(aux_filename).is_file():
            os.remove(aux_filename)
        pdf_filename = tex_filename[:-4]+'.pdf'
        png_filename = tex_filename[:-4]+'.png'
        if not os.path.exists(pdf_filename):
            output_err(output_path, 0, 'cannot compile', img_path)
        else:
            os.system("convert -density 200 -quality 100 %s %s"%(pdf_filename, png_filename))
            os.remove(pdf_filename)
            if os.path.exists(png_filename):
                crop_image(png_filename, output_path)
                os.remove(png_filename)

        
if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
