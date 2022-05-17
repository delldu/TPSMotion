CUDA_VISIBLE_DEVICES=0 python demo.py \
	--config config/vox-256.yaml \
	--checkpoint checkpoints/vox.pth.tar \
	--source_image feynman.jpeg \
	--driving_video 2.mp4
