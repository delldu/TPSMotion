CUDA_VISIBLE_DEVICES=0 python demo.py \
	--config config/vox-256.yaml \
	--checkpoint checkpoints/vox.pth.tar \
	--source_image images/0001.png \
	--driving_video videos/0006.mp4
