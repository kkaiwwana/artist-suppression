from pathlib import Path
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

if __name__ == '__main__':
    repo_root = Path.cwd().resolve()
    if repo_root.name == 'demo_notebook':
        repo_root = repo_root.parent
    
    HF_REPO_ID = 'amaai-lab/JamendoMaxCaps'
    METADATA_DIR = repo_root / 'datasets' / 'jamendo_max_caps' / 'metadata'
    OUTPUT_DIR = repo_root / 'datasets' / 'jamendo_max_caps' / 'analysis'
    METADATA_RE = re.compile(r'^\d{4}-\d{2}-\d{2}\.jsonl$')
    
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f'Project root: {repo_root}')
    print(f'Metadata directory: {METADATA_DIR}')
    
    api = HfApi()
    repo_files = api.list_repo_files(HF_REPO_ID, repo_type='dataset')
    metadata_repo_files = sorted(
        path for path in repo_files
        if '/' not in path and METADATA_RE.fullmatch(path)
    )
    
    if not metadata_repo_files:
        raise RuntimeError(
            '未找到日期型 JSONL metadata；请检查网络连接或数据集文件结构是否已变化。'
        )
    
    print(f'发现 {len(metadata_repo_files):,} 个 metadata 文件。')
    print('范围:', metadata_repo_files[0], '->', metadata_repo_files[-1])
    
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        local_dir=METADATA_DIR,
        allow_patterns='final_caption30sec.jsonl',
    )
    
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type='dataset',
        local_dir=METADATA_DIR,
        allow_patterns=metadata_repo_files,
    )
    

    
    
    metadata_files = [METADATA_DIR / name for name in metadata_repo_files]
    missing_files = [path for path in metadata_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f'下载后仍缺少 {len(missing_files)} 个文件，例如 {missing_files[0].name}')
    
    downloaded_mib = sum(path.stat().st_size for path in metadata_files) / 1024**2
    print(f'下载完成：{len(metadata_files):,} 个文件，共 {downloaded_mib:,.1f} MiB。')