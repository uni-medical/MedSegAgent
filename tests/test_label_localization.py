"""Every executable model label has a display translation as the registry evolves."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from medsegagent.catalog import native_labels, public_targets, public_task_names
from medsegagent.labels import label_display_names

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/medsegagent/label_names.json"


def test_canonical_display_names_cover_exactly_the_public_native_and_composite_labels():
    names = json.loads(SOURCE.read_text(encoding="utf-8"))
    available = set().union(
        *(native_labels(task).values() for task in public_task_names()),
        *(public_targets(modality) for modality in ("CT", "MR")),
    )
    assert set(names) == available
    for name, translations in names.items():
        assert set(translations) == {"zh", "en"}
        assert label_display_names(name) == {
            "display_name_zh": translations["zh"],
            "display_name_en": translations["en"],
        }


def test_feedback_label_names_do_not_guess_or_expose_mutable_shared_metadata():
    for name in ("future_model_region", "自定义区域", "constructor", "__proto__", "", None):
        assert label_display_names(name) == {}
    displayed = label_display_names("heart")
    assert displayed == {"display_name_zh": "心脏整体", "display_name_en": "Whole heart"}
    displayed["display_name_zh"] = "changed"
    assert label_display_names("heart")["display_name_zh"] == "心脏整体"


def test_generated_browser_dictionary_is_up_to_date():
    subprocess.run(
        [sys.executable, str(ROOT / "ops/generate_label_names.py"), "--check"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )


def test_browser_uses_the_same_canonical_bilingual_names_as_model_feedback():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser label dictionary check")
    names = json.loads(SOURCE.read_text(encoding="utf-8"))
    source = ROOT / "src/medsegagent/web_static/label_names.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const window = {};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), { window });
const labels = window.MedSegLabels;
const names = JSON.parse(fs.readFileSync(0, 'utf8'));
const missing = Object.keys(names).filter(name => !labels.has(name));
assert.deepEqual(missing, [], 'New registry labels need reviewed display translations');
for (const [name, expected] of Object.entries(names)) {
  const zh = labels.displayName(name, 1, 'zh-CN');
  const en = labels.displayName(name, 1, 'en');
  assert.equal(zh, expected.zh, name);
  assert.equal(en, expected.en, name);
  assert.match(zh, /\p{Script=Han}/u, name);
  assert.doesNotMatch(zh, /[A-Za-z_]/, name);
  assert.match(en, /[A-Za-z]/, name);
  assert.doesNotMatch(en, /\p{Script=Han}|_/u, name);
}
for (const locale of ['zh-CN', 'en']) {
  assert.equal(labels.displayName('自定义区域', 7, locale), '自定义区域');
  assert.equal(labels.displayName('', 7, locale), '7');
}
assert.equal(labels.displayName('future_model_region', 7, 'zh-CN'), 'future_model_region');
assert.equal(labels.displayName('lower_left_central_incisor_pulp_fdi131', 7, 'zh-CN'), '左下中切牙牙髓（31）');
assert.equal(labels.displayName('vertebrae_L6', 7, 'zh-CN'), '第 6 腰椎');
assert.equal(labels.displayName('rib_right_12', 7, 'zh-CN'), '右第 12 肋');
"""
    subprocess.run(
        [node, "-e", script, str(source)],
        input=json.dumps(names),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
