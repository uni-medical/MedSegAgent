# Third-party notices

Licenses and attribution for the software, model weights and example images used by
TotalSeg Agent.

## Software and weights

- TotalSegmentator 2.18.0: [Apache-2.0 code license](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/LICENSE).
  This configuration includes 32 publicly available tasks from the publisher's Apache-2.0
  group and the separately licensed `brain_aneurysm` task. See the pinned
  [publisher's list](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/README.md) and
  [prepared model manifest](../ops/open-model-manifest.json) for the exact models and sources.
- Brain aneurysm segmentation: [published model](https://doi.org/10.5281/zenodo.17894703),
  [associated paper](https://doi.org/10.1007/s10278-025-01533-3),
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Included for noncommercial
  use on TOF MRI. Preserve the publisher's attribution and license when sharing these
  weights. Result metadata retains the task's license and acquisition requirements.
  The 18 license-gated tasks and two unpublished/test tasks are excluded from preparation.
- NiiVue 0.69.0: BSD-2-Clause; the vendored license is retained with the viewer assets.
- Transitive packages retain their own licenses.

## Example images

Three images are installed separately in the example library. Each example provides
its original source link and a downloadable notice containing full attribution,
original license text, data-use policy and any transformation description. Source-download
responses also include a `Link` header with `rel="license"`. Preserve these notices when
redistributing the data.

| Example | Source and attribution | License and changes |
| --- | --- | --- |
| Abdominal CT | Jakob Wasserthal (2026), [Zenodo version 20272572](https://doi.org/10.5281/zenodo.20272572), case `SzFnYhqzTtnDAHXq` | Publisher record: CC BY 4.0; archive metadata separately declares Apache-2.0. Both declarations are retained. Original image bytes unchanged. |
| Chest CT | Armato III et al. (2015), [LIDC-IDRI / TCIA](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX), case `LIDC-IDRI-0001` | CC BY 3.0. Converted to NIfTI and resampled to 2.5 mm with unchanged outer field of view. Small structures may change. |
| Limited-field abdominal MRI | Jakob Wasserthal and contributors, [bundled test image at v2.18.0](https://github.com/wasserth/TotalSegmentator/blob/v2.18.0/tests/reference_files/example_mr_sm.nii.gz) | Official repository Apache-2.0 license; original bytes unchanged. Attribution applies to this bundled file. Through-plane coverage is 60 mm. |

The chest example retains TCIA's complete downloaded notice, data citation and
[data-use policies](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/),
including attribution, downstream preservation and no re-identification or subject contact.
The MRI example's full Apache-2.0 text is included in the downloaded notice.

To install prepared examples:

```bash
uv run python ops/install_examples.py --catalog /path/to/candidate-catalog.json \
  --assets /path/to/prepared --destination runtime/examples
```

Installation checks image validity and SHA256 and preserves the three fixed example IDs.
The installed catalog is `runtime/examples/manifest.json`.
