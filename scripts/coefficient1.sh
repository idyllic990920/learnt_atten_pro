# 0.1proto-0.1atten-1label-1ms, 1proto-1atten-1label
python cnn_mlp_adver.py --global_epoch 10 --total_num 4500 \
                        --lamda_proto 0.1 --lamda_atten 0.1 --lamda_label 1 --lamda_ms 1 \
                        --beta_proto 1 --beta_atten 1 --beta_label 1 --log_name './log/msloss/coeff1'
# 0.1proto-1atten-0.1label-1ms, 1proto-1atten-1label
python cnn_mlp_adver.py --global_epoch 10 --total_num 4500 \
                        --lamda_proto 0.1 --lamda_atten 1 --lamda_label 0.1 --lamda_ms 1 \
                        --beta_proto 1 --beta_atten 1 --beta_label 1 --log_name './log/msloss/coeff1'
