"""Generate the synchronous browser dictionary from the canonical JSON source.

To update anatomical display names, edit src/medsegagent/label_names.json, then run:
    python ops/generate_label_names.py
    python -m pytest tests/test_label_localization.py

Only the JSON is maintained by hand. Commit it together with the generated
web_static/label_names.js. Use --check to detect an outdated browser artifact.
The generator uses only the Python standard library and performs no network I/O.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "medsegagent"
SOURCE = PACKAGE / "label_names.json"
TARGET = PACKAGE / "web_static" / "label_names.js"

PREFIX = """/* Generated from src/medsegagent/label_names.json. Do not edit by hand.
 * Update the JSON, then run: python ops/generate_label_names.py
 * Display names only: upstream label identifiers, masks and files are unchanged.
 */
(() => {
  "use strict";

  const names = new Map(Object.entries({
"""
SUFFIX = """  }));

  window.MedSegLabels = Object.freeze({
    has: (name) => names.has(name),
    displayName(name, id, locale) {
      const translated = names.get(name);
      if (translated) return locale === "en" ? translated.en : translated.zh;
      // Unknown/user-defined labels retain their identity instead of guessing anatomy.
      if (locale === "en" && typeof name === "string" && /^[a-z][a-z0-9_]*$/.test(name)) {
        const label = name.replaceAll("_", " ");
        return label.charAt(0).toUpperCase() + label.slice(1);
      }
      return name || String(id);
    },
  });
})();
"""


def render() -> str:
    names = json.loads(SOURCE.read_text(encoding="utf-8"))
    if not isinstance(names, dict) or not names:
        raise ValueError("Expected a non-empty label dictionary")
    rows = []
    for name, translations in sorted(names.items()):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(translations, dict)
            or set(translations) != {"zh", "en"}
            or any(not isinstance(value, str) or not value.strip() for value in translations.values())
        ):
            raise ValueError(f"Invalid translations for label {name!r}")
        # Fix locale order to keep generation deterministic even after JSON formatting.
        value = {"zh": translations["zh"], "en": translations["en"]}
        rows.append(f"    {json.dumps(name)}: {json.dumps(value, ensure_ascii=False)},\n")
    return PREFIX + "".join(rows) + SUFFIX


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the generated JS is outdated")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != expected:
            print("Label dictionary is outdated. Run: python ops/generate_label_names.py")
            return 1
        print("Label dictionary is up to date.")
        return 0
    TARGET.write_text(expected, encoding="utf-8")
    print(f"Generated {TARGET.relative_to(PACKAGE.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
