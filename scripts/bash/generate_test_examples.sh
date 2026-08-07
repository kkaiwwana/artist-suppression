python scripts/generate_artist_classifier_audio.py \
    --checkpoint "logs/0728_091705@condition_learning_with_pretrained_adapter_1e-4lr/ckpts/epoch002-step1035.ckpt" \
    --musicgen-model "models/musicgen-small" \
    --continuation \
    --audio-prompt-seconds 5 \
    --continuation-seconds 10 \
    --groups-per-artist 2 \
    --duration-seconds 15 \
    --batch-size 8 \
    --precision fp32