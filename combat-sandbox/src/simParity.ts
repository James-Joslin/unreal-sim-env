export type Vec2 = [number, number];

/** C++/Python movement contract: 0=stay, 1-8 rotate clockwise in the target-facing frame. */
export function movementIndexToWorldDir(
  moveIndex: number,
  selfPosition: Vec2,
  targetPosition: Vec2 | null,
  fallbackFacing: Vec2,
): Vec2 {
  if (moveIndex <= 0 || moveIndex > 8) return [0, 0];

  let forward: Vec2 = fallbackFacing;
  if (targetPosition) {
    forward = [
      targetPosition[0] - selfPosition[0],
      targetPosition[1] - selfPosition[1],
    ];
  }

  const forwardLength = Math.hypot(forward[0], forward[1]);
  if (forwardLength <= 1e-8) return [0, 0];
  forward = [forward[0] / forwardLength, forward[1] / forwardLength];

  // Matches FVector::CrossProduct(Forward, UpVector) and Python [fy, -fx].
  const right: Vec2 = [forward[1], -forward[0]];
  const angle = (moveIndex - 1) * Math.PI / 4;
  return [
    forward[0] * Math.cos(angle) + right[0] * Math.sin(angle),
    forward[1] * Math.cos(angle) + right[1] * Math.sin(angle),
  ];
}

/** Segment sweep against an already-expanded AABB, including UE-style initial overlap handling. */
export function sweepAabbT(
  px: number, py: number, dx: number, dy: number,
  ox: number, oy: number, hw: number, hh: number,
): number | null {
  const bx1 = ox - hw, by1 = oy - hh, bx2 = ox + hw, by2 = oy + hh;
  let tmin = 0, tmax = 1;

  for (const [p, d, lo, hi] of [[px, dx, bx1, bx2], [py, dy, by1, by2]]) {
    if (Math.abs(d) < 1e-4) {
      if (p < lo || p > hi) return null;
    } else {
      let t1 = (lo - p) / d, t2 = (hi - p) / d;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) return null;
    }
  }

  const startsPenetrating = px > bx1 && px < bx2 && py > by1 && py < by2;
  if (tmin <= 1e-8 && !startsPenetrating) {
    // A tangent body moving away is not an initial overlap. The plain slab
    // test reports t=0 for both entering and leaving, so sample just ahead.
    const sampleX = px + dx * 1e-6;
    const sampleY = py + dy * 1e-6;
    const entersBox = sampleX > bx1 && sampleX < bx2 && sampleY > by1 && sampleY < by2;
    if (!entersBox) return null;
  }
  return Math.max(tmin, 0);
}
