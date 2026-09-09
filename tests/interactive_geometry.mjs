import assert from "node:assert/strict";
import test from "node:test";
import {worldToVoxel, voxelToWorld, boxPrompt, canvasPoint, prepareSubmission} from "../src/medsegagent/web_static/interactive.mjs";

const affines = [
  [[2, 0, 0, -10], [0, 3, 0, 15], [0, 0, 4, 30], [0, 0, 0, 1]],
  [[0, -2, 0, 60], [3, 0, 0, -15], [0, 0, -4, 30], [0, 0, 0, 1]],
  [[1.3, -0.7, 0.2, -10], [0.8, 2.1, 0.3, -20], [0.1, 0.4, 4, 7], [0, 0, 0, 1]],
  [[0.002, 0, 0, -0.1], [0, 0.003, 0, 0.05], [0, 0, 0.004, 0], [0, 0, 0, 1]],
];
test("world coordinates round trip into original unequal XYZ for flips, permutations, oblique affines and meters", () => {
  for (const affine of affines) {
    const voxel = [5.1, 7.2, 11.3];
    const actual = worldToVoxel(affine, voxelToWorld(affine, voxel));
    actual.forEach((value, k) => assert.ok(Math.abs(value - voxel[k]) < 1e-8));
  }
});
test("native box accepts all axis orientations and snaps half-slice centers", () => {
  for (const affine of affines) for (const fixed of [0, 1, 2]) {
    const axes = [0, 1, 2].filter((k) => k !== fixed);
    const corners = [[2, 3], [6, 3], [6, 8], [2, 8]].map(([a, b]) => {
      const p = [0, 0, 0]; p[fixed] = 10.5; p[axes[0]] = a; p[axes[1]] = b;
      return voxelToWorld(affine, p);
    });
    const result = boxPrompt(corners, {affine, shape: [20, 30, 40]});
    assert.equal(result.kind, "box");
    const native = worldToVoxel(affine, result.world_start);
    assert.ok(Math.abs(native[fixed] - 11) < 1e-8);
  }
});
test("box rejects diagonal-only illusions, degenerate and out-of-grid gestures", () => {
  const affine = affines[0], geometry = {affine, shape: [20, 30, 40]};
  for (const corners of [
    [[2, 2, 10], [4, 6, 10], [8, 8, 10], [6, 4, 10]],
    [[2, 3, 10], [2.1, 3, 10], [2.1, 8, 10], [2, 8, 10]],
    [[-3, 3, 10], [6, 3, 10], [6, 8, 10], [-3, 8, 10]],
    [[2, 3, 10], [6, 3, 11], [6, 8, 12], [2, 8, 11]],
  ]) assert.throws(() => boxPrompt(corners.map((p) => voxelToWorld(affine, p)), geometry));
});
test("CSS offsets and high DPI map to the actual NiiVue drawing-buffer pixels", () => {
  const canvas = {width: 1600, height: 1200, getBoundingClientRect: () => ({left: 200, top: 100, width: 800, height: 600})};
  assert.deepEqual(canvasPoint({clientX: 400, clientY: 250}, canvas), [400, 300]);
});

test("lost submission ACK reuses identical body and id after server head advances", () => {
  const body = {operation: "refine", prompts: [{world: [1, 2, 3]}], base_revision: "old", expected_revision: "old"};
  const pending = prepareSubmission(body, null, () => "request-1");
  const replay = prepareSubmission({...body, expected_revision: "already-created"}, pending,
    () => { throw new Error("must not issue a new request id"); });
  assert.equal(replay, pending);
  assert.equal(replay.body.expected_revision, "old");
  const changed = prepareSubmission({...body, prompts: [{world: [4, 5, 6]}]}, pending, () => "request-2");
  assert.equal(changed.body.message_id, "request-2");
});
