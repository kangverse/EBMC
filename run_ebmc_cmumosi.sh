# CMUMOSI

python -u EBMC/train_EBMC.py --dataset=CMUMOSI --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=64 --epochs=300 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.55 --attn_drop_rate=0.0 --test_condition=atv --stage_epoch=150 --gpu=0 --lambda_msd=0.5 --lambda_imtd=0.1 --lambda_cce=0.1 --lambda_emc=0.1
