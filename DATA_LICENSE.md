# Data license and attribution

This file says which parts of this repository have which license. It is a plain-language notice, not legal advice. Details about the data are in [docs/data.md](docs/data.md).

## Summary

| Material | Owner | License | In this repository? |
|---|---|---|---|
| MVTec AD images and masks | MVTec Software GmbH | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) | **No.** Not committed (`/data/` is git-ignored). |
| Original project code | The project owner | **No `LICENSE` file exists and none is planned** (owner decision, 2026-11-11), so the code has no licence granting reuse rights | Yes |
| Third-party libraries and pretrained weights | Their authors | Their own licenses | Not copied here; installed through `uv`. The pretrained backbone weights are downloaded by the fit step and are used; their recorded terms are in [docs/data.md](docs/data.md). |

## The dataset: MVTec AD

- **Name:** MVTec Anomaly Detection dataset (MVTec AD).
- **Creators and copyright holder:** Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, MVTec Software GmbH. The dataset's own `readme.txt` says: "Copyright 2019 MVTec Software GmbH".
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. Text: <https://creativecommons.org/licenses/by-nc-sa/4.0/>. The dataset's own `readme.txt` states this license and says that use "that falls under the commercial use clause" requires contacting MVTec.
- **Official page:** <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>. It states the dataset's license terms: reported as checked by an AI assistant (ChatGPT) and relayed by the project owner on 2026-11-10. Claude could not open the page (the connection failed), so it has not checked the page itself. The license statement in this file comes from the `readme.txt` and `license.txt` inside the local copy.

### Where the local copy came from

The project owner says the local copy (`data/raw/mvtec-ad/`) was downloaded from **Kaggle**. The specific Kaggle page, and any terms shown on it, are **unconfirmed**. This file therefore relies on the license text bundled inside the dataset, which is MVTec's own. The copy has not been compared file by file with MVTec's official archive (only the per-category image counts match the official paper's table, and that table was supplied by the owner and not checked by Claude against the paper). That the copy is identical to the official archive is **[Unverified]**.

### What this means for this project

- **Non-commercial.** This repository is public and the project is a non-commercial educational demo. It must not be used to make money. Anyone reusing the dataset, or anything derived from it, must follow the license themselves.
- **The dataset is not redistributed.** No MVTec image or mask is committed. Each person who wants to reproduce the work downloads the dataset and puts it under `data/raw/mvtec-ad/`.
- **Derived material.** Things made from the data (heatmaps, overlays, resized or re-encoded images, sample galleries, and any model fitted on it) are adaptations. If they are shared, they keep the attribution below, say they were modified, and are shared under the same license terms for the dataset-derived part (ShareAlike). Local sample sheets and histograms are written only to `data/processed/`, which is git-ignored. **Since 2026-11-10 the folder `evaluation/` also holds such material:** the `figures/*.png` sheets contain MVTec AD images, resized and overlaid with heatmaps and outlines. They are adaptations of the dataset and are under CC BY-NC-SA 4.0, not under the project's code license, with the attribution below; the images were modified. The JSON, CSV and log files there contain only numbers, file names and hashes. **These figures are already public:** they are tracked in Git and have been on the public GitHub repository since 2026-11-10 (checked 2026-11-11). They keep this notice. Whether the notice is sufficient, and whether their publication needs MVTec's confirmation, is **unresolved**; see [docs/licensing-review.md](docs/licensing-review.md).
- **Model files.** The fitted model files (`artifacts/`, git-ignored) hold feature vectors computed from MVTec AD images. The project treats them as material derived from the dataset and applies the same terms as a precaution (attribution, non-commercial, share-alike for the derived part); this is a cautious policy, not a legal finding. Each packaged artifact carries a `NOTICE.md` with the attribution and terms, whose hash is recorded in its manifest, so the notice travels with the weights.
- **Statistics files.** `splits/bottle.json` lists file names and hashes, and `experiments/*.json` hold counts and summary numbers. They contain no image content.
- **No endorsement.** Nothing here implies that MVTec Software GmbH or the authors support or endorse this project.

## Attribution (to cite when using the dataset)

From the dataset's own `readme.txt`, verbatim:

> If you use the dataset in scientific work, please cite:
>
> Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
> "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
> IEEE Conference on Computer Vision and Pattern Recognition, 2019

The journal version of the dataset paper (found by an AI assistant (ChatGPT) on Springer's publisher page and relayed by the project owner; Claude copied it and could not open the page itself):

> Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *International Journal of Computer Vision*, 129, 1038–1059. <https://doi.org/10.1007/s11263-020-01400-4>

See [docs/data.md](docs/data.md).

## Original project code

The project code currently has no licence granting reuse rights: no `LICENSE` file exists, and the owner has decided not to add one (2026-11-11). The repository is public and viewable, but that does not mean the owner grants a general licence to copy, modify or redistribute the code. This does not change the terms for the dataset, the figures or the model weights described in this file.
