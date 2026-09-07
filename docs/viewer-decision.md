# Web viewer decision

Checked against official project releases, tagged source, and npm metadata on **2026-09-07**. These observations are dated; release channels can change.

## Choice: NiiVue 0.69.0

| Candidate | Version observed | License | Fit for this application |
| --- | --- | --- | --- |
| NiiVue | npm `latest` and the official repository's latest release both identify **0.69.0**, published 2026-05-20 | BSD-2-Clause | Native NIfTI / gzipped NIfTI, WebGL2, multiplanar slices, volume rendering, discrete label lookup tables, browser and touch interaction. A single self-hosted UMD bundle works without a JavaScript application framework or build pipeline. |
| Cornerstone3D | Official GitHub latest release **5.8.9**, published 2026-09-03; npm `@cornerstonejs/core` `latest` returned **5.8.2** at inspection | MIT | Supports NIfTI through the separate `@cornerstonejs/nifti-volume-loader`, alongside a broad radiology rendering and tools framework. Appropriate for a larger DICOM / annotation workstation; introduces extra core, loader, viewport and tools setup for this NIfTI-only workflow. |

The version-channel discrepancy for Cornerstone is recorded rather than treating GitHub and npm as interchangeable. NiiVue's repository announces a transition to `niivue/mono` and a 1.0 release candidate. This implementation deliberately pins the verified 0.69.0 release; a future migration should repeat geometry, overlays and browser acceptance.

NiiVue is an image viewer, not a segmentation model or a statement of medical-device suitability. MedSegAgent remains research use only.

Official references:

- [NiiVue 0.69.0 release](https://github.com/niivue/niivue/releases/tag/%40niivue/niivue-v0.69.0)
- [NiiVue npm metadata](https://registry.npmjs.org/@niivue/niivue/0.69.0)
- [NiiVue tagged README and supported formats](https://github.com/niivue/niivue/blob/%40niivue/niivue-v0.69.0/README.md)
- [NiiVue tagged BSD license](https://github.com/niivue/niivue/blob/%40niivue/niivue-v0.69.0/LICENSE)
- [NiiVue API documentation](https://niivue.com/docs/api/niivue/classes/Niivue/)
- [Cornerstone3D 5.8.9 release](https://github.com/cornerstonejs/cornerstone3D/releases/tag/v5.8.9)
- [Cornerstone npm core metadata](https://registry.npmjs.org/@cornerstonejs/core/5.8.2)
- [Cornerstone tagged NIfTI loader](https://github.com/cornerstonejs/cornerstone3D/tree/v5.8.9/packages/nifti-volume-loader)
- [Cornerstone tagged MIT license](https://github.com/cornerstonejs/cornerstone3D/blob/v5.8.9/LICENSE)

## Distribution and reproducibility

The application serves `src/medsegagent/web_static` under `/static`. It makes no requests to a viewer CDN, analytics endpoint, remote font host, or external image host. NIfTI requests use same-origin session authentication. The script is copied unchanged from the official npm tarball, not rebuilt from a moving branch.

| Artifact | Verification |
| --- | --- |
| Official tarball | `https://registry.npmjs.org/@niivue/niivue/-/niivue-0.69.0.tgz` |
| Tarball SHA-256 | `6e9a6a471b428eb3b258a0086beca9187d95e43a781a39e8a519c6da246af485` |
| Vendored `dist/niivue.umd.js` bytes | 2,284,381 |
| Vendored bundle SHA-256 | `47b896b77ec4a5be3ef1949c33ad393b6f45629921f326c279000bcf51bdb4af` |

`vendor/NiiVue-LICENSE.txt` preserves the original BSD license. `vendor/THIRD-PARTY-NOTICES.txt` retains the bundled dependency licenses. Dependency names are confirmed by the published `dist/metafile-esm.json`, versions come from the official tagged `package-lock.json`, and license texts come from the exact dependency tarballs. Optional embedded native-codec notices are also included using the numcodecs build-script versions. The NIfTI UI does not invoke the Zarr / native-codec paths.

## Interaction and data boundaries

The work area pairs a compact upload and natural-language form with a large image canvas. On a narrow screen it becomes a vertical form, viewer, results, and task history. A local Avenir/PingFang/system font stack, cool neutral surfaces and one blue control accent keep the interface focused on images. Label colors encode only segmentation IDs.

- Uploads accept one `.nii` or `.nii.gz`. The browser checks extension, nonzero size and the server-provided compressed upload limit before transmitting. Authoritative validation of decompressed size, geometry and voxel contents is the backend's responsibility.
- Raw upload bodies use `application/octet-stream`. `X-Filename` is percent-encoded with `encodeURIComponent`; the server decodes it once and applies filename validation. This allows non-ASCII filenames without invalid HTTP headers.
- The access token is submitted once to `/api/session`, cleared from the input, and never placed in local storage, session storage, URLs or viewer headers. An HttpOnly same-origin cookie authenticates subsequent requests. Server-side authorization, CSRF checks and secure cookie attributes remain mandatory.
- Task IDs alone are stored in the page's `?task=` URL to restore selection after a refresh. All task and file reads must still enforce owner authorization. The browser does not persist image bytes or task contents.
- Polling reads persisted task status; progress is either the actual server percentage or an indeterminate bar. A connection loss does not mark an active server task as failed. The browser keeps a message ID when retrying an uncertain create request.
- Task changes serialize NiiVue scene loads. An obsolete request cannot replace the current scene; the previous canvas is hidden while another task's image loads. Logout clears the scene and in-memory state.
- Original and segmentation volumes use the same geometry. Segmentation lookup tables are discrete; label toggles set table alpha without modifying or renumbering voxel data. Nearest interpolation preserves label boundaries. The initial crosshair uses the largest nonempty label's approximate foreground centroid transformed through the NIfTI affine.
- The opacity control changes only overlay alpha. Source intensities remain visible with all labels hidden. Result downloads remain available if WebGL2 fails. A context-loss error explains that the server task remains available.
- WebGL2 and sufficient browser RAM/VRAM are required. A compressed file under the upload cap may still be too large to display on a phone. Server-side expanded-data limits and the context-loss UI are essential; successful rendering of one sample is not a universal hardware guarantee.

## Repeatable checks

From the repository root:

```sh
node --check src/medsegagent/web_static/app.js
uv run python - <<'PY'
import hashlib
from pathlib import Path
from html.parser import HTMLParser
import re

root = Path('src/medsegagent/web_static')
class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = []; self.assets = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        for key in ('src', 'href'):
            if attrs.get(key, '').startswith('/static/'):
                self.assets.append(attrs[key][len('/static/'):])
p = Page(); p.feed((root / 'index.html').read_text())
assert len(p.ids) == len(set(p.ids)), 'duplicate HTML IDs'
refs = set(re.findall(r"\$\('([^']+)'\)", (root / 'app.js').read_text()))
assert refs <= set(p.ids), refs - set(p.ids)
assert all((root / asset).is_file() for asset in p.assets)
bundle = (root / 'vendor/niivue-0.69.0.umd.js').read_bytes()
assert hashlib.sha256(bundle).hexdigest() == '47b896b77ec4a5be3ef1949c33ad393b6f45629921f326c279000bcf51bdb4af'
print('Static references, JavaScript IDs, and pinned viewer hash verified')
PY
```

Browser acceptance must additionally exercise a real server and authenticated task. The useful sequence is: login; upload a known NIfTI; submit natural language; inspect progress; inspect original and mask in slices and 3D; hide an individual label and all labels; change opacity; download both artifacts; refresh at a task URL; inspect failure and cancellation; reject an upload over the configured cap; logout; verify the task/file endpoints reject an unauthenticated session. Repeat at desktop and mobile widths and inspect canvas screenshots, not just HTTP responses or the presence of a canvas element.

Independent frontend check on 2026-09-07 used a temporary, local synthetic fixture service (not committed, not a model inference): a 64 × 64 × 48 NIfTI with 2 × 2 × 3 mm spacing and two asymmetric label regions. Chromium rendered source intensities, both colored overlays and 3D with hardware WebGL2 at 1440 × 1000 and a 390 × 844 mobile viewport. Refresh restored the task and its overlays; hiding all labels removed the colors; a file one byte over the fixture's 4 MiB cap was rejected before upload; a persisted failed task displayed its public error. Browser local/session storage remained empty and the HttpOnly cookie was unreadable through `document.cookie`. After replacing an invalid initial color-map name, the verified reload produced no JavaScript errors or NiiVue warnings. This fixture check establishes frontend behavior only; real inference, production authentication, retention, restart and public-endpoint evidence belong in the deployment acceptance record.
