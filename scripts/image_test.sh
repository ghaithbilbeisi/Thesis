src_dir=../data/im2text/images
src_test=../data/im2text/src-test.txt
tg_test=../data/im2text/tgt-test.txt
pred_log=$1 #../results/logs/pred_log1.txt
save_result=$2 #../results/predictions/results1.txt
test_list=../data/im2text/test_filter.lst
labels_norm=../data/im2text/formulas.norm.lst
#labels_orig = $9
#render_dir = $10

#python3 ../ONMT/translate.py -data_type img -model $model -src_dir $src_dir \
#	                    -src $src_test -tgt $tg_test -output predictions.txt\
#			                        -max_length 150 -beam_size 5 -gpu 0 --batch_size 5 -verbose >$pred_log 2>&1

#python3 pred_proc.py $pred_log $src_test $save_result
#bleu=$(python3 evaluation/evaluate_bleu.py --result-path $save_result --data-path $test_list --label-path $labels_norm)
#edist=$(python3 evaluation/evaluate_text_edit_distance.py --result-path $save_result)
#echo $bleu
#echo $edist

python3 evaluation/render_latex.py --result-path $save_result --data-path $test_list --label-path ../data/Original_labels/im2latex_formulas.lst --output-dir ../results/images_rendered --replace
python3 evaluation/evaluate_image.py --images-dir ../results/images_rendered/
