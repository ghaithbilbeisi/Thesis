mv ../ONMT/onmt/encoders/image_encoder.py ../ONMT/onmt/encoders/image_encoder_og.py
mv ../ONMT/onmt/encoders/image_encoder_3.py ../ONMT/onmt/encoders/image_encoder.py
sh test.sh ../results/models/im2latex-28_step_200000.pt ../results/logs/pred_log28_200.txt ../results/predictions/results28_200.txt
sh test.sh ../results/models/im2latex-28_step_90000.pt ../results/logs/pred_log28_90.txt ../results/predictions/results28_90.txt
mv ../ONMT/onmt/encoders/image_encoder.py ../ONMT/onmt/encoders/image_encoder_3.py
mv ../ONMT/onmt/encoders/image_encoder_4.py ../ONMT/onmt/encoders/image_encoder.py
sh test.sh ../results/models/im2latex-29_step_100000.pt ../results/logs/pred_log29_100.txt ../results/predictions/results29_100.txt
sh test.sh ../results/models/im2latex-29_step_90000.pt ../results/logs/pred_log29_90.txt ../results/predictions/results29_90.txt
sh test.sh ../results/models/im2latex-29_step_80000.pt ../results/logs/pred_log29_80.txt ../results/predictions/results29_80.txt
mv ../ONMT/onmt/encoders/image_encoder.py ../ONMT/onmt/encoders/image_encoder_4.py
mv ../ONMT/onmt/encoders/image_encoder_og.py ../ONMT/onmt/encoders/image_encoder.py
sh test.sh ../results/models/im2latex-31_step_100000.pt ../results/logs/pred_log31_100.txt ../results/predictions/results31_100.txt
sh test.sh ../results/models/im2latex-31_step_30000.pt ../results/logs/pred_log31_30.txt ../results/predictions/results31_30.txt
