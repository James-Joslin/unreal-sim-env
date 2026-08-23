import assert from "node:assert/strict";
import { movementIndexToWorldDir, sweepAabbT } from "./simParity.ts";
import type { Vec2 } from "./simParity.ts";

function near(actual: Vec2, expected: Vec2) {
  assert.ok(Math.abs(actual[0] - expected[0]) < 1e-6, `${actual[0]} != ${expected[0]}`);
  assert.ok(Math.abs(actual[1] - expected[1]) < 1e-6, `${actual[1]} != ${expected[1]}`);
}

// Facing east: forward, right, backward, left.
near(movementIndexToWorldDir(1, [0, 0], [100, 0], [0, 1]), [1, 0]);
near(movementIndexToWorldDir(3, [0, 0], [100, 0], [0, 1]), [0, -1]);
near(movementIndexToWorldDir(5, [0, 0], [100, 0], [0, 1]), [-1, 0]);
near(movementIndexToWorldDir(7, [0, 0], [100, 0], [0, 1]), [0, 1]);

// Rotating the target rotates the same action frame.
near(movementIndexToWorldDir(1, [0, 0], [0, 100], [1, 0]), [0, 1]);
near(movementIndexToWorldDir(5, [0, 0], [0, 100], [1, 0]), [0, -1]);
near(movementIndexToWorldDir(0, [0, 0], [100, 0], [1, 0]), [0, 0]);

// A 30 UU body touching the right face of a 50 UU half-width wall.
assert.equal(sweepAabbT(80, 0, 100, 0, 0, 0, 80, 130), null);
assert.equal(sweepAabbT(80, 0, -100, 0, 0, 0, 80, 130), 0);
assert.equal(sweepAabbT(79.9, 0, 100, 0, 0, 0, 80, 130), 0);

console.log("sim parity checks passed");
