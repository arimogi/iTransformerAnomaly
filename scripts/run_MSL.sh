echo "[START]=====================> run on: `date`"

export CUDA_VISIBLE_DEVICES=0

python run.py \
  --is_training 0 \
  --model_id MSL-test \
  --model iTransformer \
  --data MSL \
  --root_path ./dataset/MSL \
  --seq_len 100 \
  --label_len 0 \
  --pred_len 100 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 55 --dec_in 55 --c_out 55 \
  --d_model 55 \
  --des 'anomaly_test' \
  --itr 1

echo "[STOP]=====================> ended on: `date`"