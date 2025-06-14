export CUDA_VISIBLE_DEVICES=0

python run.py \
  --is_training 0 \
  --model_id SMD-test \
  --model iTransformer \
  --data SMD \
  --root_path ./dataset/SMD \
  --seq_len 100 \
  --label_len 0 \
  --pred_len 100 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 38 --dec_in 38 --c_out 38 \
  --des 'anomaly_test' \
  --itr 1

