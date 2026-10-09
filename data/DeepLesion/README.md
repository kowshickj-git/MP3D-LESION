# data/DeepLesion

DeepLesion parts 01-04, laid out the way the configs and the web app expect.
The files are too large for the repository and are attached to the
[v1.0 release](https://github.com/kowshickj-git/MP3D-LESION/releases/tag/v1.0);
RUNNING.md, "Downloads", has the commands.

| Path | Contents | From the release |
|---|---|---|
| `annotation/` | COCO-style annotations and the `_local` subsets | `DeepLesion_annotation.zip` |
| `DL_info.csv` | NIH's per-lesion metadata (pixel spacing, coarse type) | `DeepLesion_annotation.zip` |
| `zips/` | NIH's `Images_png_01.zip` to `Images_png_04.zip` | `Images_png_0N.zip.001` to `.009`, joined |
| `Images_png/Images_png/` | the extracted CT slices | made from `zips/` by `tools/prepare_deeplesion.py` |

`run_project.bat` fetches only what the web app needs: the annotations and the
slices around its 32 built-in samples (`DeepLesion_sample_slices.zip`).
