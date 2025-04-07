mkdir -p checkpoints 
huggingface-cli download Wan-AI/Wan2.1-T2V-14B "Wan2.1_VAE.pth" --local-dir checkpoints
echo "Wan2.1_VAE.pth downloaded"