# Selective Music Unlearning project page

This is a dependency-free static project page. The repository's GitHub Pages
workflow publishes this directory without moving or rebuilding it.

## Preview locally

From the repository root:

```powershell
D:\conda\python.exe -m http.server 8000 --directory project_page
```

Open <http://localhost:8000>. Nothing is uploaded by this command.

## Publish with GitHub Pages

The workflow in `.github/workflows/pages.yml` deploys `project_page/` whenever
the `ICASSP_27` branch changes and can also be started manually from the Actions
page. Before the first deployment, set the repository to public and select
**GitHub Actions** under **Settings → Pages → Build and deployment → Source**.

After committing the page and workflow to `ICASSP_27`, follow the **Deploy
project page** run under the Actions tab. The expected public URL is
<https://kkaiwwana.github.io/artist-suppression/>.

## Manage listening examples by hand

The page always reads this self-contained folder:

```text
project_page/demo_data/
├── manifest.json
├── display_order.json
└── audio/
```

To replace or update the examples, edit or replace `manifest.json`, place every
referenced audio file under `audio/`, and refresh the page. No page rebuild is
required. A minimal template is provided in `manifest.example.json`.

Two manifest layouts are recognized:

1. The original `paper_demo_bundle/manifest.json` layout, using either `title`
   values such as `demo single` and `demo multiple` or `note` values such as
   `demo_single` and `demo_multiple`, plus its existing `audio` mapping.
2. The compact generated layout currently stored here, using an explicit
   `group` and six normalized `variants`.

For original manifests, the page automatically selects the correct
single-target or multiple-target suppression fields. Waveforms are calculated
in the browser when precomputed peaks are not present.

To hide a sample without deleting its JSON block, either set `"visible": false`
on that item or add its id to the top-level `excluded_sample_ids` list.

## Configure listening examples

Edit `demo_data/display_order.json`. Each entry shows a `sample_index` and a
human-readable title.

- Move a complete sample line to reorder the page.
- Set that line's `show` field to `false` to hide the sample without deleting
  its audio or manifest entry.
- With `include_unlisted_samples: true`, newly added manifest items appear
  automatically after the explicitly ordered items. Set it to `false` to use
  the `single` and `multiple` lists as strict allowlists; in that mode, deleting
  a line also removes that sample from the page.

Refresh the page after editing. The title is only a note; matching uses
`sample_index`.

The **Show Suppress Others** switch is part of the page itself. It is off by
default for every visit and reveals the extra non-target row immediately; no
JSON edit or rebuild is needed.

The order file is separate from the generated manifest, so rebuilding the
audio bundle does not overwrite manual curation.

## Enter quantitative results

Edit `results_data/main_results.json`. Replace a `null` value with either a
number or a preformatted string. Numbers use the metric's configured decimal
precision; `null` is rendered as an em dash.

To reproduce the paper's emphasis, add a metric id to that row's `highlights`
object with either `"best"` (bold) or `"second"` (underlined):

```json
"values": { "clap": 0.314, "fad": 2.417 },
"highlights": { "clap": "best", "fad": "second" }
```

## Regenerate the curated folder from the evaluation bundle

```powershell
D:\conda\python.exe .\scripts\build_project_page_assets.py
```

The builder selects Prompt, Reference (GT), No Control, Enhance, Suppress
Target, and Suppress Others. Existing top-level exclusions are preserved when
the generated manifest is rebuilt. Pass `--clear-exclusions` to restore every
source-manifest item.

Adding an example does not require page-code changes. Add its JSON entry and
referenced audio files to `evaluation_outputs/paper_demo_bundle/`, then run the
builder command above. With `include_unlisted_samples: true`, it appears at the
end automatically; otherwise add one readable line for its `sample_index` to
`display_order.json`.

During a rebuild, `display_order.json` is synchronized automatically. Existing
manual order and `show` values are preserved, removed manifest items are
cleaned up, and new items are appended to their group when
`include_unlisted_samples` is `true`.

## Update per-sample MERT similarity

The listening cards read `demo_data/mert_similarity.json` and show each audio
variant's cosine similarity to its sample's Reference (GT). The established
evaluation runtime uses the final layer of `m-a-p/MERT-v1-95M` with
attention-mask-aware temporal mean pooling.

After adding or changing demo audio, run:

```powershell
D:\conda\python.exe .\scripts\compute_project_page_mert_similarity.py --device cuda
```

The first run downloads the MERT model. Later runs compare audio fingerprints
and encode only new or changed samples; unchanged recorded values are reused.
