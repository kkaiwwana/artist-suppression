# Learning Post-hoc Controls For Artist Suppression

## Quick Start
### Dataset Preparation
**Step 1** Download metadata from HF with `python scripts/download_metadata.py`.

**Step 2** Download a subset of JamendoMaxCaps with following command:
```python
python scripts/download_jamendomaxcaps_subset.py \
  --num artists 256 \
  --songs-per-artist 64 \
  --download-workers 32 \
  --ffmpeg-workers 8
```
Note: Dataset wukk eventually takes ~80GB disk space. So make sure 160GB space are available at least. Downloading typically takes ~4hrs.

**Step 3** Pre-process dataset with MusicGen Encodec. The encoded token format is compatible with all MusicGen series. At first, you will need a MusicGen model to use its Encodec. Simply, you can manually download it from [https://huggingface.co/facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) and move all files to the folder `models/musicgen-small`. Finally, the command is:
```python
python scripts/convert_subset_to_encodec.py \
  --subset-dir datasets/jamendo_max_caps/subset/signature_artists0256_songs064_seed0042 \
  --device cuda \
  --batch-size 8 \
  --num-workers 4 \
  --local-files-only
```
Note: Ideally, the script will makedir `datasets/jamendo_max_caps` in step.1 and download metadata at `jamendo_max_caps/metadata`, then script in step.2 will download the subset at `datasets/jamendo_max_caps/subset/signature_[ARTIST_NUMBER]_[SONG_NUMBER]_[RAMDOM_SEED]` then the encodec will output the data at same directory.

### Training and Evaluation
**Step 0** Please configure your wandb (or swanlab; an alternative) api key at `swanlab_api_key.txt / wandb_api_key.txt`; or you can configure it as env varibles. We will use them to track the experiment.

**Step 1.A** Simply finetune MusicGen to adapt to our subset with LoRA adapter with command `python scripts/run.py runner=gen_fintuning exp.cmt=[YOUR_COMMENT_HERE]`. We only apply a very light 1-epoch-training, and please check the path of your checkpoint, which can be found at dir `logs/[TIMES & DATES]@[YOUR COMMENT]/ckpts`. Configure your path to the adapter in the `config/runners/model/unlearnable_musicgen.yaml`.

**Step 1.B** Indepently train a artist classifier with command `python scripts/run.py runner=train_classifieir exp.cmt=[YOUR_COMMENT_HERE]`. Again, copy the checkpoint path and configure it at `config/runners/condition_learning.yaml`.

**Step 2** Step.1 (both A and B) can quickly be done in 30mins and now we can start real training. Very simply use `python scripts/run.py runner=condition_learning exp.cmt=[YOUR_COMMENT_AGAIN!]`. Default parameters are fine, but you may need alter some of them to reproduce the reported results in different scenario (e.g., override the training branch config with specifying `runner.model.control_learning.branch_cycle` as `[positive, positive, positive, positive, preservation]` to enable specificity regularization once in five training steps). The experiment will be tracked in W&B or Swanlab, depending on the logging backend you choose (by default, we use Swanlab; it's faster.)

## Citation
```
bib will be available on our paper release!
```
