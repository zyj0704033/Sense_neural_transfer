python ./neural_style/neural_style.py train --dataset /mnt/lustre/zhangyujing/face_data/meitu_2016  --style-image /mnt/lustre/zhangyujing/Open_Source_Code/examples/fast_neural_style/styles/mean_pose/Michael\ D.\ Edens.jpg  --save-model-dir ./snapshot/  --epochs 2 --cuda 1 --batch-size 16 --style-weight 1e10 --log-interval 200