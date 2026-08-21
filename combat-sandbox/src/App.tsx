import React, { useState, useEffect, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";
import { RewardD3Chart, RewardStepLog } from "./components/RewardD3Chart";
import { ActionProbabilityHeatmap } from "./components/ActionProbabilityHeatmap";
import { ObservationGroupInspector } from "./components/ObservationGroupInspector";
import { RewardBudgetBar } from "./components/RewardBudgetBar";
import { BatchEpisodeRunner, EpisodeResult, BatchResults } from "./components/BatchEpisodeRunner";

// Configure WASM paths for onnxruntime-web
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

// ═══════════════════════════════════════════════════════════════════
//  Constants (match C++ / Python exactly)
// ═══════════════════════════════════════════════════════════════════
const OBS_SIZE = 249;
const FRAME_STACK = 3;
const MOVEMENT_ACTIONS = 9;
const COMBAT_ACTIONS = 9;
const TARGET_ACTIONS = 5;
const DT = 0.2; // Python CombatEnvConfig.decision_interval
const S = 0.7071067811865476; // 1/√2
const AGENT_BODY_RADIUS = 30;
const DEFENCE_CONSTANT = 100;
const MIN_DAMAGE = 1;
const MAX_ACCELERATION = 2048;
const BRAKING_DECELERATION = 2048;
// These are display-only compass glyphs. Policy movement is target-relative;
// use movementDirection() for simulation semantics.
const MOVE_DIRS = [[0, 0], [1, 0], [S, -S], [0, -1], [-S, -S], [-1, 0], [-S, S], [0, 1], [S, S]];
const SPATIAL_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315];
const LOCK_NAMES = ["", "Firing", "Reloading", "Dodging", "Melee", "Switching", "WindUp", "Reposition"];
const COMBAT_NAMES = ["None", "Fire", "Reload", "Sw0", "Sw1", "Melee", "Block", "Dodge", "Reposition"];

// ═══════════════════════════════════════════════════════════════════
//  Weapon Presets (match Python WEAPON_PRESETS["heavy"])
// ═══════════════════════════════════════════════════════════════════
interface WeaponPreset {
  slots: {
    name: string;
    baseDmg: number;
    range: number;
    maxAmmo: number;
    fireCd: number;
    reloadTime: number;
    windUp?: number;
    projSpeed: number;
    optMin?: number;
    optMax?: number;
    maxArcHeight?: number;
    canArc: boolean;
  }[];
  melee: { damage: number; range: number; cooldown: number };
}

const WEAPON_PRESETS: Record<string, WeaponPreset> = {
  scout: {
    slots: [
      { name: "Laser", baseDmg: 8, range: 1200, maxAmmo: 20, fireCd: 0.2, reloadTime: 2.0, projSpeed: 4500, optMin: 400, optMax: 900, canArc: false },
    ],
    melee: { damage: 15, range: 150, cooldown: 0.8 },
  },
  heavy: {
    slots: [
      { name: "Cannon", baseDmg: 35, range: 2000, maxAmmo: 6, fireCd: 1.0, reloadTime: 3.0, windUp: 0.5, projSpeed: 2000, optMin: 800, optMax: 1600, canArc: false },
      { name: "Missiles", baseDmg: 25, range: 1800, maxAmmo: 4, fireCd: 1.5, reloadTime: 4.0, projSpeed: 1200, optMin: 600, optMax: 1400, maxArcHeight: 400, canArc: true },
    ],
    melee: { damage: 40, range: 250, cooldown: 1.5 },
  },
  sniper: {
    slots: [
      { name: "Railgun", baseDmg: 80, range: 3000, maxAmmo: 1, fireCd: 2.0, reloadTime: 3.0, windUp: 1.0, projSpeed: 6000, optMin: 1500, optMax: 2800, canArc: false },
      { name: "Sidearm", baseDmg: 10, range: 1000, maxAmmo: 12, fireCd: 0.3, reloadTime: 2.0, projSpeed: 3500, optMin: 300, optMax: 800, canArc: false },
    ],
    melee: { damage: 10, range: 150, cooldown: 1.0 },
  },
  melee_bot: {
    slots: [
      { name: "Sidearm", baseDmg: 8, range: 800, maxAmmo: 10, fireCd: 0.4, reloadTime: 2.0, projSpeed: 3000, optMin: 200, optMax: 600, canArc: false },
    ],
    melee: { damage: 35, range: 200, cooldown: 0.6 },
  },
  tank: {
    slots: [
      { name: "Gatling", baseDmg: 5, range: 1500, maxAmmo: 100, fireCd: 0.08, reloadTime: 4.0, projSpeed: 4000, optMin: 400, optMax: 1200, canArc: false },
    ],
    melee: { damage: 30, range: 250, cooldown: 1.2 },
  },
};
// Types
export interface Weapon {
  name: string;
  baseDmg: number;
  range: number;
  maxAmmo: number;
  fireCd: number;
  reloadTime: number;
  windUp?: number;
  projSpeed: number;
  optMin?: number;
  optMax?: number;
  maxArcHeight?: number;
  canArc: boolean;
  ammo: number;
  cdRemain: number;
  reloadRemain: number;
  isReloading: boolean;
}

export interface Obstacle {
  x: number;
  y: number;
  hw: number;
  hh: number;
  height: number;
}

export interface Target {
  id: number;
  pos: [number, number];
  vel: [number, number];
  facing: [number, number];
  hp: number;
  maxHp: number;
  alive: boolean;
  role: string; // combat role: ranged/melee/mixed
  isPlayerControlled: boolean;
  characterType: number;
  mana: number;
  maxMana: number;
  commitment: number;
  commitmentDuration: number;
  commitmentTimer: number;
  gapCloserRange: number;
  gapCloserCd: number;
  hasGapCloser: boolean;
  maxSpeed: number;
  behaviour: string;
  strafeDir: number;
  strafeTimer: number;
  moveTimer: number;
  moveDir: [number, number];
  defence: number;
  barrier: number;
  atkCd: number;
  atkCooldown: number;
  projSpeed: number;
  atkDmg: number;
  atkRange: number;
  atkStat: number;
  meleeDmg: number;
  meleeRange: number;
  meleeCooldown: number;
  meleeCd: number;
  meleeStat: number;
  attackManaCost: number;
  manaRegen: number;
  manaRegenDelay: number;
  manaRegenDelayRemain: number;
  critChance: number;
  critMult: number;
}

export interface AllyState {
  id: number;
  pos: [number, number];
  vel: [number, number];
  facing: [number, number];
  hp: number;
  maxHp: number;
  defence: number;
  alive: boolean;
  archetype: number;
  maxSpeed: number;
  attackRange: number;
  attackDamage: number;
  attackCooldown: number;
  attackCd: number;
  targetId: number;
  combatAction: number;
}

export interface Projectile {
  pos: [number, number];
  vel: [number, number];
  damage: number;
  atkStat: number;
  critChance: number;
  critMult: number;
  isAgent: boolean;
  isPlayer?: boolean;
  canArc: boolean;
  life: number;
  ownerId?: number;
  arcStart?: [number, number];
  arcApex?: [number, number];
  arcEnd?: [number, number];
  arcElapsed?: number;
  arcFlightTime?: number;
  arcImpactRadius?: number;
}

export interface AgentState {
  pos: [number, number];
  vel: [number, number];
  facing: [number, number];
  hp: number;
  maxHp: number;
  barrier: number;
  defence: number;
  atkStat: number;
  critChance: number;
  critMult: number;
  weapons: Weapon[];
  activeWeapon: number;
  melee: { damage: number; range: number; cooldown: number; cdRemain: number };
  isDodging: boolean;
  dodgeRemain: number;
  dodgeCd: number;
  dodgeDir: [number, number];
  dodgeDuration: number;
  dodgeCooldown: number;
  isRepositioning: boolean;
  repositionRemain: number;
  repositionCd: number;
  repositionDir: [number, number];
  repositionDuration: number;
  repositionCooldown: number;
  repositionSpeedMultiplier: number;
  isSwitching: boolean;
  switchRemain: number;
  switchTarget: number;
  switchTime: number;
  isWindingUp: boolean;
  windUpRemain: number;
  pendingFire: { targetPos: [number, number]; targetVel: [number, number]; slotIdx: number; targetId: number } | null;
  lockRemain: number;
  lockDuration: number;
  lockReason: number;
  combatTime: number;
  maxSpeed: number;
  maxAcceleration: number;
  brakingDeceleration: number;
  cachedMovementAction: number;
  isBlocking: boolean;
  blockDefenceBonus: number;
  blockMovementMultiplier: number;
  spawnPos: [number, number];
  leashRange: number;
  activeTargetIdx: number; // raw index into sim.targets, matching Python current_target_idx
  threatTable?: Record<number, number>;
}

export interface PlayerState {
  pos: [number, number];
  facing: [number, number];
  hp: number;
  maxHp: number;
  defence: number;
  atkStat: number;
  critChance: number;
  critMult: number;
  weapon: {
    name: string;
    baseDmg: number;
    maxAmmo: number;
    ammo: number;
    fireCd: number;
    cdRemain: number;
    reloadTime: number;
    reloadRemain: number;
    isReloading: boolean;
    projSpeed: number;
    range: number;
  };
}

export interface SimState {
  arena: number;
  half: number;
  preset: WeaponPreset;
  agent: AgentState;
  targets: Target[];
  allies: AllyState[];
  obstacles: Obstacle[];
  projectiles: Projectile[];
  player: PlayerState;
  interactivePlayer: boolean;
  status: { stunRemain: number; slowRemain: number; slowStrength: number; debuffs: { remain: number; strength: number }[] };
  playerPatterns: { aggression: number; evasion: number; predictability: number; preferredRange: number; manaBurn: number };
  playerPatternHist: number[];
  targetActionSlots: number[];
  targetSlotScores: Record<number, number>;
  targetSlotHps: Record<number, number>;
  targetSlotLos: Record<number, boolean>;
  targetSnapshotElapsed: number;
  curriculumStage: number;
  agentArchetype: number;
  rngState: number;
  // Authoritative per-transition attribution, mirroring Python's damage event ledger.
  stepAgentDamageByTarget: Record<number, number>;
  stepAgentDamageTaken: number;
  stepAgentKillIds: number[];
  step: number;
  maxSteps: number;
  done: boolean;
}

// Math and Utility Functions
const dist = (a: [number, number], b: [number, number]): number =>
  Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);

const norm = (v: [number, number]): [number, number] => {
  const d = Math.sqrt(v[0] ** 2 + v[1] ** 2) || 1;
  return [v[0] / d, v[1] / d];
};

const dot = (a: [number, number], b: [number, number]): number =>
  a[0] * b[0] + a[1] * b[1];

const clamp = (x: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, x));

function nextRng(state: number): [number, number] {
  let x = (state >>> 0) || 0x6d2b79f5;
  x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
  x >>>= 0;
  return [x, x / 0x100000000];
}

function simRandom(sim: SimState): number {
  const [state, value] = nextRng(sim.rngState);
  sim.rngState = state;
  return value;
}

function simRand(sim: SimState, lo: number, hi: number): number {
  return lo + simRandom(sim) * (hi - lo);
}

function segPointDist(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  if (len2 < 0.01) return Math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2);
  const t = clamp(((cx - ax) * dx + (cy - ay) * dy) / len2, 0, 1);
  return Math.sqrt((cx - (ax + t * dx)) ** 2 + (cy - (ay + t * dy)) ** 2);
}

function computeDamage(baseDmg: number, atkStat: number, defence: number, barrier = 0, critChance = 0, critMult = 1.5, random: () => number = Math.random) {
  let outgoing = baseDmg + atkStat;
  const wasCrit = random() < critChance;
  if (wasCrit) outgoing *= critMult;

  const barrierAbsorbed = Math.min(barrier, outgoing);
  const remaining = outgoing - barrierAbsorbed;
  const newBarrier = barrier - barrierAbsorbed;
  if (remaining <= 0) return { hpDamage: 0, newBarrier, wasCrit };

  // Python: remaining * 100 / (defence + 100), with a 1 HP floor.
  const hpDamage = Math.max(remaining * DEFENCE_CONSTANT / (defence + DEFENCE_CONSTANT), MIN_DAMAGE);
  return { hpDamage, newBarrier, wasCrit };
}

function rayAABB(px: number, py: number, dx: number, dy: number, ox: number, oy: number, hw: number, hh: number): number | null {
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
  return tmin;
}

function checkLOS(a: [number, number], b: [number, number], obstacles: Obstacle[]): boolean {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  for (const o of obstacles) {
    if (rayAABB(a[0], a[1], dx, dy, o.x, o.y, o.hw, o.hh) !== null) return false;
  }
  return true;
}

function pushOutAABB(pos: [number, number], obstacles: Obstacle[], radius = AGENT_BODY_RADIUS): [number, number] {
  for (const o of obstacles) {
    const cx = clamp(pos[0], o.x - o.hw, o.x + o.hw);
    const cy = clamp(pos[1], o.y - o.hh, o.y + o.hh);
    const dx = pos[0] - cx, dy = pos[1] - cy;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < radius && d > 0.01) {
      pos[0] = cx + dx / d * radius;
      pos[1] = cy + dy / d * radius;
    } else if (d < 0.01) {
      pos[0] = o.x + o.hw + radius;
    }
  }
  return pos;
}

function cloneSimState(sim: SimState): SimState {
  return {
    ...sim,
    agent: {
      ...sim.agent, pos: [...sim.agent.pos], vel: [...sim.agent.vel], facing: [...sim.agent.facing],
      dodgeDir: [...sim.agent.dodgeDir], repositionDir: [...sim.agent.repositionDir], spawnPos: [...sim.agent.spawnPos],
      weapons: sim.agent.weapons.map(w => ({ ...w })), melee: { ...sim.agent.melee },
      pendingFire: sim.agent.pendingFire ? { ...sim.agent.pendingFire, targetPos: [...sim.agent.pendingFire.targetPos], targetVel: [...sim.agent.pendingFire.targetVel] } : null,
      threatTable: { ...(sim.agent.threatTable || {}) },
    },
    targets: sim.targets.map(t => ({ ...t, pos: [...t.pos], vel: [...t.vel], facing: [...t.facing], moveDir: [...t.moveDir] })),
    allies: sim.allies.map(a => ({ ...a, pos: [...a.pos], vel: [...a.vel], facing: [...a.facing] })),
    status: { ...sim.status, debuffs: sim.status.debuffs.map(d => ({ ...d })) },
    playerPatterns: { ...sim.playerPatterns }, playerPatternHist: [...sim.playerPatternHist],
    obstacles: sim.obstacles.map(o => ({ ...o })),
    projectiles: sim.projectiles.map(pr => ({ ...pr, pos: [...pr.pos], vel: [...pr.vel], arcStart: pr.arcStart ? [...pr.arcStart] : undefined, arcApex: pr.arcApex ? [...pr.arcApex] : undefined, arcEnd: pr.arcEnd ? [...pr.arcEnd] : undefined })),
    player: { ...sim.player, pos: [...sim.player.pos], facing: [...sim.player.facing], weapon: { ...sim.player.weapon } },
    targetActionSlots: [...sim.targetActionSlots],
    targetSlotScores: { ...sim.targetSlotScores },
    targetSlotHps: { ...sim.targetSlotHps },
    targetSlotLos: { ...sim.targetSlotLos },
    stepAgentDamageByTarget: { ...sim.stepAgentDamageByTarget },
    stepAgentKillIds: [...sim.stepAgentKillIds],
  };
}

function currentTarget(sim: SimState): Target | null {
  const idx = sim.agent.activeTargetIdx;
  return idx >= 0 && idx < sim.targets.length ? sim.targets[idx] : null;
}

function targetSnapshotIsValid(sim: SimState): boolean {
  const aliveIds = new Set(sim.targets.filter(t => t.alive).map(t => t.id));
  if (!sim.targetActionSlots.length) return aliveIds.size === 0;
  return sim.targetActionSlots.every(id => aliveIds.has(id));
}

function publishTargetSnapshot(sim: SimState) {
  const scored = sim.targets.filter(t => t.alive).map(t => ({
    target: t,
    // Python adds 0..10 priority jitter when publishing the immutable slots.
    score: scoreTarget(sim, t) + simRandom(sim) * 10,
  }));
  scored.sort((a, b) => b.score - a.score || a.target.id - b.target.id);
  const top = scored.slice(0, TARGET_ACTIONS - 1);
  sim.targetActionSlots = top.map(x => x.target.id);
  sim.targetSlotScores = {};
  sim.targetSlotHps = {};
  sim.targetSlotLos = {};
  for (const item of top) {
    const t = item.target;
    sim.targetSlotScores[t.id] = item.score;
    sim.targetSlotHps[t.id] = t.hp / Math.max(t.maxHp, 1);
    sim.targetSlotLos[t.id] = checkLOS(sim.agent.pos, t.pos, sim.obstacles);
  }
  sim.targetSnapshotElapsed = 0;
}

function ensureTargetSnapshot(sim: SimState) {
  if (!targetSnapshotIsValid(sim)) publishTargetSnapshot(sim);
}

function getSortedTargets(sim: SimState): Target[] {
  ensureTargetSnapshot(sim);
  const byId = new Map(sim.targets.filter(t => t.alive).map(t => [t.id, t]));
  return sim.targetActionSlots.map(id => byId.get(id)).filter((t): t is Target => !!t);
}

function movementDirection(sim: SimState, moveIdx: number): [number, number] {
  if (moveIdx <= 0 || moveIdx >= MOVEMENT_ACTIONS) return [0, 0];
  const ag = sim.agent;
  const target = currentTarget(sim);
  let fwd: [number, number] = [1, 0];
  if (target?.alive) {
    const toTarget: [number, number] = [target.pos[0] - ag.pos[0], target.pos[1] - ag.pos[1]];
    if (Math.hypot(toTarget[0], toTarget[1]) > 1) fwd = norm(toTarget);
  }
  const right: [number, number] = [fwd[1], -fwd[0]];
  const angle = (moveIdx - 1) * Math.PI / 4;
  return norm([
    fwd[0] * Math.cos(angle) + right[0] * Math.sin(angle),
    fwd[1] * Math.cos(angle) + right[1] * Math.sin(angle),
  ]);
}

function executeMovement(sim: SimState, moveIdx: number, dt: number) {
  const a = sim.agent;
  let movementMaxSpeed = a.maxSpeed;
  if (sim.status.slowRemain > 0) movementMaxSpeed *= Math.max(0, 1 - sim.status.slowStrength);
  if (a.isBlocking) movementMaxSpeed *= a.blockMovementMultiplier;
  if (a.isRepositioning) movementMaxSpeed *= a.repositionSpeedMultiplier;

  const dir = movementDirection(sim, moveIdx);
  const desired: [number, number] = [dir[0] * movementMaxSpeed, dir[1] * movementMaxSpeed];
  const diff: [number, number] = [desired[0] - a.vel[0], desired[1] - a.vel[1]];
  const diffMag = Math.hypot(diff[0], diff[1]);
  if (diffMag < 0.01) {
    a.vel = [...desired];
  } else {
    const currentSpeed = Math.hypot(a.vel[0], a.vel[1]);
    const desiredSpeed = Math.hypot(desired[0], desired[1]);
    const accel = desiredSpeed < 1 || desiredSpeed < currentSpeed ? a.brakingDeceleration : a.maxAcceleration;
    const maxChange = accel * dt;
    if (diffMag <= maxChange) a.vel = [...desired];
    else a.vel = [a.vel[0] + diff[0] / diffMag * maxChange, a.vel[1] + diff[1] / diffMag * maxChange];
  }

  const speed = Math.hypot(a.vel[0], a.vel[1]);
  if (speed > movementMaxSpeed && speed > 0) a.vel = [a.vel[0] / speed * movementMaxSpeed, a.vel[1] / speed * movementMaxSpeed];

  const totalDelta: [number, number] = [a.vel[0] * dt, a.vel[1] * dt];
  const moveDist = Math.hypot(totalDelta[0], totalDelta[1]);
  const substeps = Math.max(1, Math.ceil(moveDist / (AGENT_BODY_RADIUS * 0.9)));
  const stepDelta: [number, number] = [totalDelta[0] / substeps, totalDelta[1] / substeps];
  const intended: [number, number] = [a.pos[0] + totalDelta[0], a.pos[1] + totalDelta[1]];
  const next: [number, number] = [...a.pos];
  for (let i = 0; i < substeps; i++) {
    next[0] += stepDelta[0]; next[1] += stepDelta[1];
    pushOutAABB(next, sim.obstacles, AGENT_BODY_RADIUS);
  }

  const pushed: [number, number] = [next[0] - intended[0], next[1] - intended[1]];
  const pushMag = Math.hypot(pushed[0], pushed[1]);
  if (pushMag > 0.1) {
    const n: [number, number] = [pushed[0] / pushMag, pushed[1] / pushMag];
    const vn = dot(a.vel, n);
    a.vel = [a.vel[0] - n[0] * vn, a.vel[1] - n[1] * vn];
  }

  const minBound = -sim.half + AGENT_BODY_RADIUS, maxBound = sim.half - AGENT_BODY_RADIUS;
  a.pos = [clamp(next[0], minBound, maxBound), clamp(next[1], minBound, maxBound)];
  const target = currentTarget(sim);
  if (target?.alive) {
    const toTarget: [number, number] = [target.pos[0] - a.pos[0], target.pos[1] - a.pos[1]];
    if (Math.hypot(toTarget[0], toTarget[1]) > 1) a.facing = norm(toTarget);
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Sim State Factory
// ═══════════════════════════════════════════════════════════════════
function createSim(presetName: string = "heavy", arenaSize = 2500, numTargets = 2, numObstacles = 4, curriculumStage = 3, interactivePlayer = true, seed = 0x12345678): SimState {
  const half = arenaSize * 0.45;
  const wp = WEAPON_PRESETS[presetName] || WEAPON_PRESETS.heavy;
  let localRngState = seed >>> 0;
  const random = () => { const [st, v] = nextRng(localRngState); localRngState = st; return v; };
  const rp = () => (random() - 0.5) * half * 1.5;

  const weapons: Weapon[] = wp.slots.map(s => ({
    ...s, ammo: s.maxAmmo, cdRemain: 0, reloadRemain: 0, isReloading: false,
  }));

  const obstacles: Obstacle[] = [];
  for (let i = 0; i < numObstacles; i++) {
    const hw = 40 + random() * 60, hh = 40 + random() * 60;
    obstacles.push({ x: rp(), y: rp(), hw, hh, height: random() > 0.4 ? 300 : 150 });
  }

  const targets: Target[] = [];
  // Represent the player as target 0 so the AI agent recognizes and fights the Green Player
  targets.push({
    id: 0,
    pos: [500, 300],
    vel: [0, 0],
    facing: [-1, 0],
    hp: 100,
    maxHp: 100,
    alive: true,
    role: "ranged",
    isPlayerControlled: true,
    characterType: 0.6, mana: 50, maxMana: 50, commitment: 0, commitmentDuration: 0, commitmentTimer: 0,
    gapCloserRange: 600, gapCloserCd: 0, hasGapCloser: true, maxSpeed: 500,
    behaviour: "aggressive", strafeDir: 1, strafeTimer: 0, moveTimer: 0, moveDir: [1, 0],
    defence: 20,
    barrier: 0,
    atkCd: 0, atkCooldown: 1.0,
    projSpeed: 1800,
    atkDmg: 15,
    atkRange: 1500,
    atkStat: 8,
    meleeDmg: 15, meleeRange: 160, meleeCooldown: 1.2, meleeCd: 0, meleeStat: 6,
    attackManaCost: 8, manaRegen: 5, manaRegenDelay: 2, manaRegenDelayRemain: 0,
    critChance: 0.05,
    critMult: 1.5,
  });

  // Spawn additional hostile red targets if requested
  for (let i = 0; i < numTargets; i++) {
    targets.push({
      id: i + 1,
      pos: [300 + rp() * 0.5, rp() * 0.5],
      vel: [0, 0],
      facing: [-1, 0],
      hp: 100,
      maxHp: 100,
      alive: true,
      role: "ranged",
      isPlayerControlled: false,
      characterType: 0.6, mana: 50, maxMana: 50, commitment: 0, commitmentDuration: 0, commitmentTimer: 0,
      gapCloserRange: 600, gapCloserCd: 0, hasGapCloser: true, maxSpeed: 500,
      behaviour: "aggressive", strafeDir: 1, strafeTimer: 0, moveTimer: 0, moveDir: [1, 0],
      defence: 30,
      barrier: 0,
      atkCd: 0, atkCooldown: 1.0,
      projSpeed: 1800,
      atkDmg: 8,
      atkRange: 1200,
      atkStat: 5,
      meleeDmg: 15, meleeRange: 160, meleeCooldown: 1.2, meleeCd: 0, meleeStat: 6,
      attackManaCost: 8, manaRegen: 5, manaRegenDelay: 2, manaRegenDelayRemain: 0,
      critChance: 0.05,
      critMult: 1.5,
    });
  }

  const sim: SimState = {
    arena: arenaSize, half, preset: wp,
    agent: {
      pos: [rp() * 0.3, rp() * 0.3], vel: [0, 0], facing: [1, 0], hp: 130, maxHp: 130,
      barrier: 0, defence: 25, atkStat: 10, critChance: 0.05, critMult: 1.5,
      weapons, activeWeapon: 0, melee: { ...wp.melee, cdRemain: 0 },
      isDodging: false, dodgeRemain: 0, dodgeCd: 0, dodgeDir: [0, 0], dodgeDuration: 0.4, dodgeCooldown: 2.0,
      isRepositioning: false, repositionRemain: 0, repositionCd: 0, repositionDir: [0, 0],
      repositionDuration: 0.6, repositionCooldown: 3.0, repositionSpeedMultiplier: 1.75,
      isSwitching: false, switchRemain: 0, switchTarget: 0, switchTime: 0.3,
      isWindingUp: false, windUpRemain: 0, pendingFire: null,
      lockRemain: 0, lockDuration: 0, lockReason: 0, combatTime: 0,
      maxSpeed: 450, maxAcceleration: MAX_ACCELERATION, brakingDeceleration: BRAKING_DECELERATION,
      cachedMovementAction: 0, isBlocking: false, blockDefenceBonus: 80, blockMovementMultiplier: 0.3,
      spawnPos: [0, 0], leashRange: 2000, activeTargetIdx: 0,
      threatTable: {},
    },
    targets, allies: [], obstacles,
    projectiles: [],
    player: {
      pos: [500, 300], facing: [-1, 0],
      hp: 100, maxHp: 100, defence: 20, atkStat: 8,
      critChance: 0.05, critMult: 1.5,
      weapon: {
        name: "Rifle", baseDmg: 15, maxAmmo: 12, ammo: 12, fireCd: 0.35, cdRemain: 0,
        reloadTime: 2.0, reloadRemain: 0, isReloading: false, projSpeed: 3500, range: 1500
      },
    },
    interactivePlayer,
    status: { stunRemain: 0, slowRemain: 0, slowStrength: 0, debuffs: Array.from({ length: 6 }, () => ({ remain: 0, strength: 0 })) },
    playerPatterns: { aggression: 0, evasion: 0, predictability: 0.5, preferredRange: 0.5, manaBurn: 0 },
    playerPatternHist: new Array(8).fill(0),
    targetActionSlots: [], targetSlotScores: {}, targetSlotHps: {}, targetSlotLos: {}, targetSnapshotElapsed: 0,
    curriculumStage, agentArchetype: curriculumStage === 1 ? 1 : 0, rngState: localRngState,
    stepAgentDamageByTarget: {}, stepAgentDamageTaken: 0, stepAgentKillIds: [],
    step: 0, maxSteps: 100000, done: false,
  };
  publishTargetSnapshot(sim);
  return sim;
}

interface CurriculumConfig { preset: string; arena: number; targets: number; obstacles: number; maxSteps: number; enemyHp: number; enemyDefence: number; targetHp: number; targetDefence: number; targetSpeedFraction: number; engagementDistance: number; }

const CURRICULUM_CONFIGS: Record<number, CurriculumConfig> = {
  1: { preset: "melee_bot", arena: 1500, targets: 1, obstacles: 0, maxSteps: 500, enemyHp: 100, enemyDefence: 20, targetHp: 100, targetDefence: 10, targetSpeedFraction: 0.0, engagementDistance: 800 },
  2: { preset: "scout", arena: 2000, targets: 1, obstacles: 0, maxSteps: 500, enemyHp: 100, enemyDefence: 20, targetHp: 150, targetDefence: 15, targetSpeedFraction: 0.0, engagementDistance: 1200 },
  3: { preset: "scout", arena: 2500, targets: 1, obstacles: 3, maxSteps: 1000, enemyHp: 120, enemyDefence: 20, targetHp: 50, targetDefence: 20, targetSpeedFraction: 0.6, engagementDistance: 1500 },
  4: { preset: "heavy", arena: 3000, targets: 1, obstacles: 8, maxSteps: 500, enemyHp: 100, enemyDefence: 20, targetHp: 75, targetDefence: 20, targetSpeedFraction: 0.8, engagementDistance: 1500 },
  5: { preset: "heavy", arena: 3000, targets: 1, obstacles: 8, maxSteps: 600, enemyHp: 200, enemyDefence: 25, targetHp: 100, targetDefence: 25, targetSpeedFraction: 0.8, engagementDistance: 1500 },
  6: { preset: "heavy", arena: 3000, targets: 1, obstacles: 12, maxSteps: 700, enemyHp: 180, enemyDefence: 30, targetHp: 150, targetDefence: 25, targetSpeedFraction: 0.9, engagementDistance: 1500 },
  7: { preset: "heavy", arena: 4000, targets: 4, obstacles: 16, maxSteps: 800, enemyHp: 500, enemyDefence: 35, targetHp: 150, targetDefence: 25, targetSpeedFraction: 1.0, engagementDistance: 2000 },
};


function makeAlly(sim: SimState, id: number): AllyState {
  const r = simRandom(sim) * 85;
  const archetype = r < 40 ? 0 : r < 70 ? 1 : 3;
  const cfg = archetype === 1
    ? { maxSpeed: 480, attackRange: 200, attackDamage: 25, attackCooldown: 0.7, hp: 120, defence: 20 }
    : archetype === 3
    ? { maxSpeed: 320, attackRange: 600, attackDamage: 8, attackCooldown: 1.0, hp: 150, defence: 30 }
    : { maxSpeed: 380, attackRange: 1000, attackDamage: 10, attackCooldown: 0.8, hp: 80, defence: 15 };
  const pos: [number, number] = [
    clamp(sim.agent.pos[0] + (simRandom(sim) * 800 - 400), -sim.half, sim.half),
    clamp(sim.agent.pos[1] + (simRandom(sim) * 800 - 400), -sim.half, sim.half),
  ];
  return {
    id, pos, vel: [0, 0], facing: [1, 0], hp: cfg.hp, maxHp: cfg.hp,
    defence: cfg.defence, alive: true, archetype,
    maxSpeed: cfg.maxSpeed, attackRange: cfg.attackRange,
    attackDamage: cfg.attackDamage, attackCooldown: cfg.attackCooldown,
    attackCd: 0, targetId: -1, combatAction: 0,
  };
}

function gapCloserThreat(t: Target, agentPos: [number, number]): number {
  if (!t.alive || !t.hasGapCloser || t.gapCloserCd > 0) return 0;
  const d = dist(t.pos, agentPos);
  return d <= t.gapCloserRange ? clamp(1 - d / Math.max(t.gapCloserRange, 1), 0, 1) : 0;
}

function tickAllies(sim: SimState, dt: number) {
  for (const ally of sim.allies) {
    if (!ally.alive) continue;
    ally.attackCd = Math.max(0, ally.attackCd - dt);
    let target: Target | null = null;
    let best = Infinity;
    for (const t of sim.targets) {
      if (!t.alive) continue;
      const d = dist(ally.pos, t.pos);
      if (d < best) { best = d; target = t; }
    }
    ally.targetId = target?.id ?? -1;
    ally.combatAction = 0;
    if (!target) { ally.vel = [0, 0]; continue; }

    const toT = norm([target.pos[0] - ally.pos[0], target.pos[1] - ally.pos[1]]);
    let v: [number, number] = [0, 0];
    if (ally.archetype === 1) {
      const scale = best > 200 ? 1 : 0.3;
      v = [toT[0] * ally.maxSpeed * scale, toT[1] * ally.maxSpeed * scale];
    } else if (ally.archetype === 3) {
      if (best > ally.attackRange * 0.7) v = [toT[0] * ally.maxSpeed * 0.6, toT[1] * ally.maxSpeed * 0.6];
      else v = [toT[1] * ally.maxSpeed * 0.3, -toT[0] * ally.maxSpeed * 0.3];
    } else {
      if (best > ally.attackRange * 0.8) v = [toT[0] * ally.maxSpeed * 0.7, toT[1] * ally.maxSpeed * 0.7];
      else if (best < ally.attackRange * 0.3) v = [-toT[0] * ally.maxSpeed * 0.5, -toT[1] * ally.maxSpeed * 0.5];
      else v = [toT[1] * ally.maxSpeed * 0.4, -toT[0] * ally.maxSpeed * 0.4];
    }
    ally.vel = v;
    ally.pos = [clamp(ally.pos[0] + v[0] * dt, -sim.half, sim.half), clamp(ally.pos[1] + v[1] * dt, -sim.half, sim.half)];
    pushOutAABB(ally.pos, sim.obstacles, AGENT_BODY_RADIUS);
    if (Math.hypot(v[0], v[1]) > 10) ally.facing = norm(v);

    if (best <= ally.attackRange && ally.attackCd <= 0 && checkLOS(ally.pos, target.pos, sim.obstacles)) {
      ally.attackCd = ally.attackCooldown;
      ally.combatAction = ally.archetype === 1 ? 5 : 1;
      const result = computeDamage(ally.attackDamage, 5, target.defence, target.barrier, 0, 1.5, () => simRandom(sim));
      target.barrier = result.newBarrier;
      target.hp = Math.max(0, target.hp - result.hpDamage);
      if (target.hp <= 0) target.alive = false;
    }
  }

  // Matches CombatEnvExtended: every live hostile has a 30% chance each tick
  // to redirect an in-range attack to one live allied robot.
  const livingAllies = sim.allies.filter(a => a.alive);
  if (!livingAllies.length) return;
  for (const t of sim.targets) {
    if (!t.alive || simRandom(sim) >= 0.3 || t.atkCd > 0) continue;
    const ally = livingAllies[Math.floor(simRandom(sim) * livingAllies.length)];
    if (dist(t.pos, ally.pos) < t.atkRange && checkLOS(t.pos, ally.pos, sim.obstacles)) {
      const result = computeDamage(t.atkDmg, t.atkStat, ally.defence, 0, t.critChance, t.critMult, () => simRandom(sim));
      ally.hp = Math.max(0, ally.hp - result.hpDamage);
      if (ally.hp <= 0) ally.alive = false;
      t.atkCd = t.atkCooldown;
    }
  }
}

function tickExtendedSystems(sim: SimState, dt: number) {
  // Existing effects tick before new effects are sampled, matching
  // CombatEnvExtended._before_transition_finalization().
  sim.status.stunRemain = Math.max(0, sim.status.stunRemain - dt);
  sim.status.slowRemain = Math.max(0, sim.status.slowRemain - dt);
  if (sim.status.slowRemain <= 0) sim.status.slowStrength = 0;
  for (const d of sim.status.debuffs) {
    d.remain = Math.max(0, d.remain - dt);
    if (d.remain <= 0) d.strength = 0;
  }

  if (sim.curriculumStage >= 3) {
    for (const t of sim.targets) {
      if (!t.alive) continue;
      const d = dist(sim.agent.pos, t.pos);
      if (t.role === "melee" && d < 300 && t.atkCd <= 0 && simRandom(sim) < 0.08) {
        sim.status.stunRemain = 0.4 + simRandom(sim) * 0.3;
      }
      if ((t.role === "ranged" || t.role === "mixed") && t.atkCd <= 0 && simRandom(sim) < 0.05) {
        sim.status.slowRemain = 1 + simRandom(sim);
        sim.status.slowStrength = 0.2 + simRandom(sim) * 0.3;
      }
      if (simRandom(sim) < 0.02 * dt) {
        let slot = sim.status.debuffs.find(x => x.remain <= 0);
        if (!slot) slot = sim.status.debuffs.reduce((a, b) => a.remain < b.remain ? a : b);
        slot.remain = 2 + simRandom(sim) * 3;
        slot.strength = 0.1 + simRandom(sim) * 0.3;
      }
    }
  }

  tickAllies(sim, dt);

  // PlayerPatternTracker parity (alpha=0.05, 8-bin movement entropy).
  const live = sim.targets.filter(t => t.alive);
  if (live.length) {
    const alpha = 0.05;
    const aggression = live.reduce((v, t) => v + (t.commitment > 0.01 ? 1 : 0), 0) / live.length;
    const prefRange = live.reduce((v, t) => v + dist(t.pos, sim.agent.pos) / Math.max(sim.arena, 1), 0) / live.length;
    const manaBurn = live.reduce((v, t) => v + (t.maxMana > 0 ? 1 - t.mana / t.maxMana : 0), 0) / live.length;
    sim.playerPatterns.aggression = sim.playerPatterns.aggression * (1 - alpha) + aggression * alpha;
    sim.playerPatterns.preferredRange = sim.playerPatterns.preferredRange * (1 - alpha) + prefRange * alpha;
    sim.playerPatterns.manaBurn = sim.playerPatterns.manaBurn * (1 - alpha) + manaBurn * alpha;
    for (const t of live) {
      if (Math.hypot(t.vel[0], t.vel[1]) > 10) {
        const angle = Math.atan2(t.vel[1], t.vel[0]);
        const bin = Math.floor((angle + Math.PI) / (Math.PI / 4)) % 8;
        sim.playerPatternHist[bin] += 1;
      }
    }
  }
  sim.playerPatternHist = sim.playerPatternHist.map(x => x * 0.99);
  const histTotal = sim.playerPatternHist.reduce((a, b) => a + b, 0);
  if (histTotal > 1e-6) {
    let entropy = 0;
    for (const count of sim.playerPatternHist) {
      const p = count / histTotal;
      if (p > 0.001) entropy -= p * Math.log(p);
    }
    sim.playerPatterns.predictability = Math.min(1, entropy / Math.log(8));
  }
}


function rand(sim: SimState, lo: number, hi: number) { return simRand(sim, lo, hi); }

function regenerateCurriculumArena(sim: SimState, baseArena: number, baseObstacles: number) {
  const effectiveArena = baseArena * rand(sim, 0.8, 1.2);
  sim.arena = effectiveArena;
  sim.half = effectiveArena * 0.45;
  const spawnHalf = effectiveArena * 0.3;
  sim.agent.pos = [rand(sim, -spawnHalf, spawnHalf), rand(sim, -spawnHalf, spawnHalf)];
  sim.agent.spawnPos = [...sim.agent.pos];

  const lo = Math.max(0, baseObstacles - Math.floor(baseObstacles / 2));
  const hiExclusive = baseObstacles + Math.floor(baseObstacles / 2) + 1;
  const count = lo + Math.floor(simRandom(sim) * Math.max(1, hiExclusive - lo));
  const placementHalf = effectiveArena * 0.4;
  sim.obstacles = [];
  for (let i = 0; i < count; i++) {
    const ox = rand(sim, -placementHalf, placementHalf), oy = rand(sim, -placementHalf, placementHalf);
    const r = simRandom(sim) * 100;
    if (r < 15) {
      const size = rand(sim, 30, 60);
      sim.obstacles.push({ x: ox, y: oy, hw: size, hh: size, height: 300 });
    } else if (r < 40) {
      if (simRandom(sim) < 0.5) sim.obstacles.push({ x: ox, y: oy, hw: rand(sim, 150, 350), hh: rand(sim, 20, 40), height: 300 });
      else sim.obstacles.push({ x: ox, y: oy, hw: rand(sim, 20, 40), hh: rand(sim, 150, 350), height: 300 });
    } else if (r < 50) {
      const hw1 = rand(sim, 80, 200), hh1 = rand(sim, 20, 40);
      sim.obstacles.push({ x: ox, y: oy, hw: hw1, hh: hh1, height: 300 });
      sim.obstacles.push({ x: ox + hw1 * (simRandom(sim) < 0.5 ? -1 : 1), y: oy, hw: rand(sim, 20, 40), hh: rand(sim, 80, 200), height: 300 });
    } else if (r < 80) {
      sim.obstacles.push({ x: ox, y: oy, hw: rand(sim, 60, 180), hh: rand(sim, 20, 50), height: rand(sim, 100, 180) });
    } else {
      sim.obstacles.push({ x: ox, y: oy, hw: rand(sim, 100, 250), hh: rand(sim, 80, 200), height: 300 });
    }
  }
}

function configureTargetForStage(t: Target, index: number, sim: SimState, cfg: CurriculumConfig, stage: number) {
  let role: string;
  if (stage <= 2) role = "ranged";
  else if (stage <= 4) role = index === 0
    ? (simRandom(sim) < 2 / 3 ? "ranged" : "melee")
    : (simRandom(sim) < 0.5 ? "ranged" : "melee");
  else {
    const r = simRandom(sim) * 100;
    role = r < 40 ? "ranged" : r < 75 ? "melee" : "mixed";
  }
  t.role = role;
  if (role === "melee") t.behaviour = "aggressive";
  else if (stage <= 2) t.behaviour = "passive";
  else if (stage <= 4) {
    const opts = ["aggressive", "kiting", "passive"];
    t.behaviour = opts[Math.floor(simRandom(sim) * opts.length)];
  } else {
    const opts = ["aggressive", "kiting", "cover_user", "passive"];
    t.behaviour = opts[Math.floor(simRandom(sim) * opts.length)];
  }
  t.strafeDir = 1; t.strafeTimer = 0; t.moveTimer = 0; t.moveDir = [1, 0];
  t.isPlayerControlled = index === 0;
  t.characterType = role === "melee" ? 0.0 : role === "mixed" ? 0.2 : 0.6;
  t.maxSpeed = 500 * cfg.targetSpeedFraction;
  const baseHp = cfg.targetHp * rand(sim, 0.8, 1.2);
  const baseDef = cfg.targetDefence * rand(sim, 0.8, 1.2);

  if (role === "melee") {
    t.hp = t.maxHp = baseHp * 1.2; t.defence = baseDef * 1.2;
    t.meleeDmg = rand(sim, 28, 40); t.meleeRange = rand(sim, 180, 250); t.meleeCooldown = rand(sim, 0.6, 1.0); t.meleeStat = rand(sim, 10, 16);
    t.atkDmg = rand(sim, 8, 12); t.atkRange = rand(sim, 600, 900); t.atkCooldown = rand(sim, 1.2, 2.0); t.projSpeed = [2000, 2500, 3000][Math.floor(simRandom(sim) * 3)]; t.atkStat = rand(sim, 3, 6);
    t.maxMana = t.mana = 0; t.attackManaCost = 0; t.hasGapCloser = true; t.gapCloserRange = rand(sim, 500, 800);
  } else if (role === "mixed") {
    t.hp = t.maxHp = baseHp; t.defence = baseDef;
    t.meleeDmg = rand(sim, 22, 32); t.meleeRange = rand(sim, 170, 220); t.meleeCooldown = rand(sim, 0.8, 1.2); t.meleeStat = rand(sim, 8, 12);
    t.atkDmg = rand(sim, 14, 20); t.atkRange = rand(sim, 900, 1300); t.atkCooldown = rand(sim, 0.9, 1.3); t.projSpeed = [1500, 1800, 2000][Math.floor(simRandom(sim) * 3)]; t.atkStat = rand(sim, 6, 10);
    t.maxMana = rand(sim, 30, 50); t.mana = t.maxMana; t.attackManaCost = rand(sim, 5, 10); t.hasGapCloser = simRandom(sim) > 0.5; t.gapCloserRange = rand(sim, 400, 600);
  } else {
    t.hp = t.maxHp = baseHp; t.defence = baseDef;
    t.meleeDmg = rand(sim, 10, 18); t.meleeRange = rand(sim, 140, 180); t.meleeCooldown = rand(sim, 1.0, 1.5); t.meleeStat = rand(sim, 4, 8);
    t.atkDmg = rand(sim, 15, 22); t.atkRange = rand(sim, 1000, 1500); t.atkCooldown = rand(sim, 0.8, 1.4); t.projSpeed = [2500, 3000, 3500, 4000, 4500][Math.floor(simRandom(sim) * 5)]; t.atkStat = rand(sim, 5, 12);
    t.maxMana = rand(sim, 40, 80); t.mana = t.maxMana; t.attackManaCost = rand(sim, 6, 12); t.hasGapCloser = false;
  }
  t.atkCd = 0; t.meleeCd = 0; t.commitment = 0; t.commitmentDuration = 0; t.commitmentTimer = 0; t.gapCloserCd = 0; t.manaRegen = 5; t.manaRegenDelay = 2; t.manaRegenDelayRemain = 0;

  const spawnScale = role === "melee" ? rand(sim, 0.4, 0.8) : rand(sim, 0.7, 1.3);
  const angle = rand(sim, 0, Math.PI * 2), spawnD = cfg.engagementDistance * spawnScale;
  t.pos = [
    clamp(sim.agent.pos[0] + Math.cos(angle) * spawnD, -sim.half, sim.half),
    clamp(sim.agent.pos[1] + Math.sin(angle) * spawnD, -sim.half, sim.half),
  ];
  t.vel = [0, 0];
  t.facing = norm([sim.agent.pos[0] - t.pos[0], sim.agent.pos[1] - t.pos[1]]);
}

function randomizeAgentWeaponsForStage(sim: SimState, stage: number) {
  if (stage < 4) return;
  for (const w of sim.agent.weapons) {
    const rangeScale = rand(sim, 0.6, 1.8);
    w.range *= rangeScale;
    if (w.optMin != null) w.optMin *= rangeScale;
    if (w.optMax != null) w.optMax *= rangeScale;
    w.fireCd *= rand(sim, 0.7, 1.3);
    w.baseDmg *= rand(sim, 0.7, 1.3);
    w.projSpeed *= rand(sim, 0.8, 1.2);
    w.reloadTime *= rand(sim, 0.8, 1.2);
  }
}

function createCurriculumSim(stage: number, interactivePlayer = true, seed = 42): SimState {
  const cfg = CURRICULUM_CONFIGS[stage] || CURRICULUM_CONFIGS[3];
  const pool = stage >= 5 ? ["heavy", "scout", "sniper", "tank"] : [cfg.preset];
  const seedProbe = { rngState: seed >>> 0 } as SimState;
  const preset = pool[Math.floor(simRandom(seedProbe) * pool.length)];
  // Stage 7 samples squad sizes 1..4 exactly like the Python curriculum.
  const targetCount = stage === 7 ? 1 + Math.floor(simRandom(seedProbe) * 4) : cfg.targets;
  const sim = createSim(preset, cfg.arena, Math.max(0, targetCount - 1), cfg.obstacles, stage, interactivePlayer, seedProbe.rngState);
  sim.maxSteps = cfg.maxSteps;
  regenerateCurriculumArena(sim, cfg.arena, cfg.obstacles);
  sim.agent.hp = sim.agent.maxHp = cfg.enemyHp;
  sim.agent.defence = cfg.enemyDefence;
  sim.agent.atkStat = 5;
  randomizeAgentWeaponsForStage(sim, stage);
  sim.targets.forEach((t, i) => configureTargetForStage(t, i, sim, cfg, stage));
  if (interactivePlayer) {
    const p0 = sim.targets[0];
    sim.player.pos = [...p0.pos]; sim.player.facing = [...p0.facing];
    sim.player.hp = sim.player.maxHp = p0.maxHp; sim.player.defence = p0.defence;
  }
  const allyCount = stage === 6 ? 1 : stage === 7 ? Math.max(0, targetCount - 1) : 0;
  for (let i = 0; i < allyCount; i++) sim.allies.push(makeAlly(sim, i));
  publishTargetSnapshot(sim);
  return sim;
}


function rotateToward(current: [number, number], desired: [number, number], maxTurn: number): [number, number] {
  const cross = current[0] * desired[1] - current[1] * desired[0];
  const dp = clamp(dot(current, desired), -1, 1);
  const diff = Math.atan2(cross, dp);
  if (Math.abs(diff) <= maxTurn) return [...desired];
  const a = Math.sign(diff) * maxTurn, c = Math.cos(a), sn = Math.sin(a);
  return norm([current[0] * c - current[1] * sn, current[0] * sn + current[1] * c]);
}

function startTargetCommitment(t: Target, duration: number) {
  t.commitmentDuration = duration;
  t.commitmentTimer = 0;
  t.commitment = 0.01;
}

function tickTargetAI(sim: SimState, t: Target, dt: number) {
  const a = sim.agent;
  t.atkCd -= dt;
  t.meleeCd -= dt;
  t.gapCloserCd = Math.max(0, t.gapCloserCd - dt);
  if (t.manaRegenDelayRemain > 0) t.manaRegenDelayRemain -= dt;
  else if (t.mana < t.maxMana) t.mana = Math.min(t.maxMana, t.mana + t.manaRegen * dt);
  if (t.commitmentDuration > 0 && t.commitmentTimer < t.commitmentDuration) {
    t.commitmentTimer += dt;
    t.commitment = Math.min(1, t.commitmentTimer / t.commitmentDuration);
  } else {
    t.commitment = 0; t.commitmentDuration = 0; t.commitmentTimer = 0;
  }

  const toAgentVec: [number, number] = [a.pos[0] - t.pos[0], a.pos[1] - t.pos[1]];
  const d = Math.hypot(toAgentVec[0], toAgentVec[1]) || 1;
  const toAgent = norm(toAgentVec);
  const perp: [number, number] = [toAgent[1], -toAgent[0]];
  t.strafeTimer -= dt;
  if (t.strafeTimer <= 0) { t.strafeDir *= -1; t.strafeTimer = rand(sim, 1, 3); }

  let v: [number, number] = [0, 0];
  if (t.maxSpeed <= 0) {
    v = [0, 0];
  } else if (t.role === "melee") {
    if (d > t.meleeRange * 2.5) v = [toAgent[0] * t.maxSpeed, toAgent[1] * t.maxSpeed];
    else if (d > t.meleeRange) {
      const q = norm([toAgent[0] * 0.8 + perp[0] * t.strafeDir * 0.2, toAgent[1] * 0.8 + perp[1] * t.strafeDir * 0.2]);
      v = [q[0] * t.maxSpeed, q[1] * t.maxSpeed];
    } else v = [perp[0] * t.strafeDir * t.maxSpeed * 0.4, perp[1] * t.strafeDir * t.maxSpeed * 0.4];
  } else if (t.role === "mixed") {
    if (t.atkCd > 0.5 && d > t.meleeRange * 3) v = [toAgent[0] * t.maxSpeed * 0.8, toAgent[1] * t.maxSpeed * 0.8];
    else if (d > t.atkRange * 0.6) v = [toAgent[0] * t.maxSpeed * 0.7, toAgent[1] * t.maxSpeed * 0.7];
    else if (d < 300) v = [-toAgent[0] * t.maxSpeed * 0.4, -toAgent[1] * t.maxSpeed * 0.4];
    else v = [perp[0] * t.strafeDir * t.maxSpeed * 0.5, perp[1] * t.strafeDir * t.maxSpeed * 0.5];
  } else if (t.behaviour === "aggressive") {
    if (d > t.atkRange * 0.7) v = [toAgent[0] * t.maxSpeed, toAgent[1] * t.maxSpeed];
    else if (d < 300) v = [-toAgent[0] * t.maxSpeed * 0.5, -toAgent[1] * t.maxSpeed * 0.5];
    else v = [perp[0] * t.strafeDir * t.maxSpeed * 0.6, perp[1] * t.strafeDir * t.maxSpeed * 0.6];
  } else if (t.behaviour === "kiting") {
    if (d < t.atkRange * 0.5) v = [-toAgent[0] * t.maxSpeed, -toAgent[1] * t.maxSpeed];
    else if (d < t.atkRange * 0.8) {
      const q = norm([-toAgent[0] * 0.3 + perp[0] * t.strafeDir * 0.7, -toAgent[1] * 0.3 + perp[1] * t.strafeDir * 0.7]);
      v = [q[0] * t.maxSpeed * 0.7, q[1] * t.maxSpeed * 0.7];
    } else v = [perp[0] * t.strafeDir * t.maxSpeed * 0.4, perp[1] * t.strafeDir * t.maxSpeed * 0.4];
  } else if (t.behaviour === "cover_user") {
    let best: Obstacle | null = null, score = -1;
    for (const o of sim.obstacles) {
      const dc = Math.hypot(t.pos[0] - o.x, t.pos[1] - o.y);
      const ca = Math.hypot(a.pos[0] - o.x, a.pos[1] - o.y);
      if (dc < 600 && ca < d && 1 / Math.max(dc, 50) > score) { best = o; score = 1 / Math.max(dc, 50); }
    }
    if (best) {
      const toC = [best.x - t.pos[0], best.y - t.pos[1]] as [number, number];
      const dc = Math.hypot(toC[0], toC[1]);
      if (dc > 80) { const q = norm(toC); v = [q[0] * t.maxSpeed * 0.8, q[1] * t.maxSpeed * 0.8]; }
      else v = [perp[0] * t.strafeDir * t.maxSpeed * 0.3, perp[1] * t.strafeDir * t.maxSpeed * 0.3];
    } else v = [perp[0] * t.strafeDir * t.maxSpeed * 0.5, perp[1] * t.strafeDir * t.maxSpeed * 0.5];
  } else {
    t.moveTimer -= dt;
    if (t.moveTimer <= 0) { const ang = rand(sim, 0, Math.PI * 2); t.moveDir = [Math.cos(ang), Math.sin(ang)]; t.moveTimer = rand(sim, 0.5, 2); }
    v = [t.moveDir[0] * t.maxSpeed * 0.6, t.moveDir[1] * t.maxSpeed * 0.6];
  }

  t.vel = v;
  t.pos = [clamp(t.pos[0] + v[0] * dt, -sim.half, sim.half), clamp(t.pos[1] + v[1] * dt, -sim.half, sim.half)];
  // Python targets are points for obstacle correction rather than using the
  // trained agent's 30 UU collision body.
  for (const o of sim.obstacles) {
    if (Math.abs(t.pos[0] - o.x) < o.hw && Math.abs(t.pos[1] - o.y) < o.hh) {
      const dx = t.pos[0] - o.x, dy = t.pos[1] - o.y;
      if (Math.abs(dx / Math.max(o.hw, 1)) > Math.abs(dy / Math.max(o.hh, 1))) t.pos[0] = o.x + (o.hw + 5) * (dx > 0 ? 1 : -1);
      else t.pos[1] = o.y + (o.hh + 5) * (dy > 0 ? 1 : -1);
    }
  }
  const desired = norm([a.pos[0] - t.pos[0], a.pos[1] - t.pos[1]]);
  t.facing = rotateToward(t.facing, desired, 2 * Math.PI * dt);

  // Melee first; a successful melee attempt consumes the target's attack for
  // this transition exactly like _target_attacks_agent().
  const nowD = dist(t.pos, a.pos);
  if ((t.role === "melee" || t.role === "mixed") && nowD <= t.meleeRange && t.meleeCd <= 0 && t.commitment <= 0) {
    t.meleeCd = t.meleeCooldown;
    startTargetCommitment(t, 0.3);
    if (!a.isDodging) {
      const result = computeDamage(t.meleeDmg, t.meleeStat, a.defence, a.barrier, t.critChance, t.critMult, () => simRandom(sim));
      a.barrier = result.newBarrier; a.hp = Math.max(0, a.hp - result.hpDamage);
      recordAgentDamageTaken(sim, result.hpDamage);
    }
    return;
  }

  const toA = norm([a.pos[0] - t.pos[0], a.pos[1] - t.pos[1]]);
  const faceDot = dot(t.facing, toA);
  if (faceDot < 0.17 || !(t.role === "ranged" || t.role === "mixed") || t.atkCd > 0 || t.commitment > 0 || nowD > t.atkRange || !checkLOS(t.pos, a.pos, sim.obstacles)) return;
  if (t.attackManaCost > 0) {
    if (t.mana < t.attackManaCost) return;
    t.mana -= t.attackManaCost; t.manaRegenDelayRemain = t.manaRegenDelay;
  }
  t.atkCd = t.atkCooldown;
  startTargetCommitment(t, 0.4);
  const flight = nowD / Math.max(t.projSpeed, 500);
  const predicted: [number, number] = [a.pos[0] + a.vel[0] * flight, a.pos[1] + a.vel[1] * flight];
  spawnProjectile(sim, t.pos, predicted,
    { projSpeed: t.projSpeed, baseDmg: t.atkDmg, range: t.atkRange, canArc: false },
    false, t.atkStat, t.critChance, t.critMult, t.id);
}

function recordAgentDamage(sim: SimState, target: Target, hpBefore: number, hpDamage: number) {
  const dealt = Math.max(0, Math.min(hpBefore, hpDamage));
  if (dealt <= 0) return;
  sim.stepAgentDamageByTarget[target.id] = (sim.stepAgentDamageByTarget[target.id] || 0) + dealt;
  if (hpBefore > 0 && target.hp <= 0 && !sim.stepAgentKillIds.includes(target.id)) sim.stepAgentKillIds.push(target.id);
}

function recordAgentDamageTaken(sim: SimState, hpDamage: number) {
  sim.stepAgentDamageTaken += Math.max(0, hpDamage);
}

// ═══════════════════════════════════════════════════════════════════
//  Sim Tick
// ═══════════════════════════════════════════════════════════════════
function tickSim(sim: SimState, action: [number, number, number], playerPos: [number, number], dt = DT, playerActions: any = {}): SimState {
  if (sim.done) return sim;
  const ag = sim.agent;
  const pl = sim.player;
  sim.stepAgentDamageByTarget = {};
  sim.stepAgentDamageTaken = 0;
  sim.stepAgentKillIds = [];
  sim.step++;

  // Keep the manually controlled party member synchronized with its target
  // record before resolving the policy action.
  const playerTarget = sim.targets.find(t => t.isPlayerControlled);
  if (sim.interactivePlayer) {
    pl.pos = [playerPos[0], playerPos[1]];
  }
  if (playerTarget && sim.interactivePlayer) {
    const oldPos: [number, number] = [...playerTarget.pos];
    playerTarget.vel = dt > 0 ? [(playerPos[0] - oldPos[0]) / dt, (playerPos[1] - oldPos[1]) / dt] : [0, 0];
    playerTarget.pos = [playerPos[0], playerPos[1]];
    playerTarget.facing = [pl.facing[0], pl.facing[1]];
    playerTarget.hp = pl.hp;
    playerTarget.alive = pl.hp > 0;
  }

  // Enforce the same masks Gym step() enforces. Invalid externally supplied
  // heads become their production-safe no-op.
  const actionMask = buildActionMask(sim);
  const requestedMove = action[0], requestedCombat = action[1], requestedTarget = action[2];
  const moveIdx = actionMask.m[requestedMove] ? requestedMove : 0;
  const combatIdx = actionMask.c[requestedCombat] ? requestedCombat : 0;
  const targetIdx = actionMask.t[requestedTarget] ? requestedTarget : TARGET_ACTIONS - 1;

  const wasLocked = ag.lockRemain > 0;
  let effectiveMove = moveIdx;
  let effectiveCombat = combatIdx;
  let effectiveTarget = targetIdx;
  if (wasLocked) {
    effectiveMove = clamp(ag.cachedMovementAction, 0, MOVEMENT_ACTIONS - 1);
    effectiveCombat = 0;
    effectiveTarget = TARGET_ACTIONS - 1;
  } else {
    ag.cachedMovementAction = moveIdx;
  }

  // Production order is target -> combat -> movement.
  if (!wasLocked && effectiveTarget < TARGET_ACTIONS - 1) {
    const selected = getSortedTargets(sim)[effectiveTarget];
    if (selected) {
      const rawIdx = sim.targets.findIndex(t => t.id === selected.id && t.alive);
      if (rawIdx >= 0) ag.activeTargetIdx = rawIdx;
    }
  }

  const target = currentTarget(sim);
  if (!wasLocked && target?.alive) {
    const slot = ag.weapons[ag.activeWeapon];
    const targetDist = dist(ag.pos, target.pos);
    const hasLOS = checkLOS(ag.pos, target.pos, sim.obstacles);

    // Block is a held stance: every non-block combat decision releases it.
    if (effectiveCombat === 6) {
      if (!ag.isBlocking) {
        ag.defence += ag.blockDefenceBonus;
        ag.isBlocking = true;
      }
    } else if (ag.isBlocking) {
      ag.defence -= ag.blockDefenceBonus;
      ag.isBlocking = false;
    }

    if (effectiveCombat === 8) {
      // Reposition is movement-head-directed. Hold+Reposition is a true no-op.
      if (effectiveMove !== 0 && !ag.isRepositioning && ag.repositionCd <= 0 && ag.lockRemain <= 0) {
        ag.isRepositioning = true;
        ag.repositionRemain = ag.repositionDuration;
        ag.repositionCd = ag.repositionCooldown;
        setLock(ag, ag.repositionDuration, 7);
      } else {
        effectiveCombat = 0;
      }
    } else if (effectiveCombat === 1 && slot && slot.cdRemain <= 0 && slot.ammo > 0 && !slot.isReloading && !ag.isWindingUp) {
      slot.ammo--;
      slot.cdRemain = slot.fireCd;
      if ((slot.windUp || 0) > 0) {
        ag.isWindingUp = true;
        ag.windUpRemain = slot.windUp || 0;
        setLock(ag, (slot.windUp || 0) + slot.fireCd, 6);
        ag.pendingFire = { targetPos: [...target.pos], targetVel: [...target.vel], slotIdx: ag.activeWeapon, targetId: target.id };
      } else {
        setLock(ag, slot.fireCd * 0.5, 1);
        if (targetDist <= slot.range && (hasLOS || slot.canArc)) {
          const flight = targetDist / Math.max(slot.projSpeed, 500);
          const predicted: [number, number] = [target.pos[0] + target.vel[0] * flight, target.pos[1] + target.vel[1] * flight];
          spawnProjectile(sim, ag.pos, predicted, slot, true, ag.atkStat, ag.critChance, ag.critMult, target.id);
        }
      }
    } else if (effectiveCombat === 2 && slot && !slot.isReloading && slot.ammo < slot.maxAmmo) {
      slot.isReloading = true;
      slot.reloadRemain = slot.reloadTime;
      setLock(ag, slot.reloadTime, 2);
      ag.isWindingUp = false;
      ag.pendingFire = null;
    } else if (effectiveCombat === 3 && ag.weapons.length > 0 && ag.activeWeapon !== 0) {
      ag.isSwitching = true;
      ag.switchRemain = ag.switchTime;
      ag.switchTarget = 0;
      setLock(ag, ag.switchTime, 5);
      ag.isWindingUp = false;
      ag.pendingFire = null;
    } else if (effectiveCombat === 4 && ag.weapons.length > 1 && ag.activeWeapon !== 1) {
      ag.isSwitching = true;
      ag.switchRemain = ag.switchTime;
      ag.switchTarget = 1;
      setLock(ag, ag.switchTime, 5);
      ag.isWindingUp = false;
      ag.pendingFire = null;
    } else if (effectiveCombat === 5 && targetDist <= ag.melee.range && ag.melee.cdRemain <= 0) {
      const hpBefore = target.hp;
      const result = computeDamage(ag.melee.damage, ag.atkStat, target.defence, target.barrier, ag.critChance, ag.critMult, () => simRandom(sim));
      target.barrier = result.newBarrier;
      target.hp = Math.max(0, target.hp - result.hpDamage);
      if (target.hp <= 0) target.alive = false;
      recordAgentDamage(sim, target, hpBefore, result.hpDamage);
      ag.melee.cdRemain = ag.melee.cooldown;
      setLock(ag, ag.melee.cooldown, 4);
    } else if (effectiveCombat === 7 && !ag.isDodging && ag.dodgeCd <= 0) {
      const speed = Math.hypot(ag.vel[0], ag.vel[1]);
      const away = norm([ag.pos[0] - target.pos[0], ag.pos[1] - target.pos[1]]);
      ag.dodgeDir = speed > 10 ? [ag.vel[0] / speed, ag.vel[1] / speed] : away;
      ag.isDodging = true;
      ag.dodgeRemain = ag.dodgeDuration;
      ag.dodgeCd = ag.dodgeCooldown;
      ag.vel = [0, 0];
      setLock(ag, ag.dodgeDuration + 0.1, 3);
    }
  }

  // Movement executes after combat so a newly-started dodge/reposition owns
  // this same decision transition, exactly like Python.
  if (ag.isDodging) {
    const delta: [number, number] = [ag.dodgeDir[0] * 800 * dt, ag.dodgeDir[1] * 800 * dt];
    const moveDist = Math.hypot(delta[0], delta[1]);
    const substeps = Math.max(1, Math.ceil(moveDist / (AGENT_BODY_RADIUS * 0.9)));
    const stepDelta: [number, number] = [delta[0] / substeps, delta[1] / substeps];
    const next: [number, number] = [...ag.pos];
    for (let i = 0; i < substeps; i++) {
      next[0] += stepDelta[0]; next[1] += stepDelta[1];
      pushOutAABB(next, sim.obstacles, AGENT_BODY_RADIUS);
    }
    ag.pos = [
      clamp(next[0], -sim.half + AGENT_BODY_RADIUS, sim.half - AGENT_BODY_RADIUS),
      clamp(next[1], -sim.half + AGENT_BODY_RADIUS, sim.half - AGENT_BODY_RADIUS),
    ];
  } else {
    executeMovement(sim, effectiveMove, dt);
  }

  // Tick weapon/runtime state after execution.
  ag.combatTime += dt;
  for (const w of ag.weapons) {
    w.cdRemain = Math.max(0, w.cdRemain - dt);
    if (w.isReloading) {
      w.reloadRemain -= dt;
      if (w.reloadRemain <= 0) { w.isReloading = false; w.reloadRemain = 0; w.ammo = w.maxAmmo; }
    }
  }
  ag.melee.cdRemain = Math.max(0, ag.melee.cdRemain - dt);
  if (ag.isWindingUp) {
    ag.windUpRemain -= dt;
    if (ag.windUpRemain <= 0) { ag.windUpRemain = 0; ag.isWindingUp = false; }
  }
  if (ag.isSwitching) {
    ag.switchRemain -= dt;
    if (ag.switchRemain <= 0) { ag.switchRemain = 0; ag.isSwitching = false; ag.activeWeapon = ag.switchTarget; }
  }
  if (ag.isDodging) {
    ag.dodgeRemain -= dt;
    if (ag.dodgeRemain <= 0) { ag.dodgeRemain = 0; ag.isDodging = false; }
  }
  ag.dodgeCd = Math.max(0, ag.dodgeCd - dt);
  if (ag.isRepositioning) {
    ag.repositionRemain -= dt;
    if (ag.repositionRemain <= 0) { ag.repositionRemain = 0; ag.isRepositioning = false; }
  }
  ag.repositionCd = Math.max(0, ag.repositionCd - dt);
  if (ag.lockRemain > 0) {
    ag.lockRemain -= dt;
    if (ag.lockRemain <= 0) { ag.lockRemain = 0; ag.lockReason = 0; }
  }

  // Deferred wind-up shot resolves after timers tick.
  if (ag.pendingFire && !ag.isWindingUp) {
    const pending = ag.pendingFire;
    ag.pendingFire = null;
    const pendingSlot = ag.weapons[pending.slotIdx];
    if (pendingSlot) {
      const d = dist(ag.pos, pending.targetPos);
      if (d <= pendingSlot.range && (checkLOS(ag.pos, pending.targetPos, sim.obstacles) || pendingSlot.canArc)) {
        const flight = d / Math.max(pendingSlot.projSpeed, 500);
        const predicted: [number, number] = [pending.targetPos[0] + pending.targetVel[0] * flight, pending.targetPos[1] + pending.targetVel[1] * flight];
        spawnProjectile(sim, ag.pos, predicted, pendingSlot, true, ag.atkStat, ag.critChance, ag.critMult, pending.targetId);
      }
    }
  }

  // Threat decays once per decision transition.
  if (!ag.threatTable) ag.threatTable = {};
  for (const id of Object.keys(ag.threatTable)) {
    ag.threatTable[+id] = Math.max(0, ag.threatTable[+id] - 5 * dt);
    if (ag.threatTable[+id] <= 0) delete ag.threatTable[+id];
  }

  // Manually-controlled player weapon is only active in the interactive view.
  // Batch evaluation uses the same scripted target loop as Python.
  const pw = pl.weapon;
  pw.cdRemain = Math.max(0, pw.cdRemain - dt);
  if (pw.isReloading) {
    pw.reloadRemain -= dt;
    if (pw.reloadRemain <= 0) { pw.isReloading = false; pw.reloadRemain = 0; pw.ammo = pw.maxAmmo; }
  }
  if (sim.interactivePlayer && playerActions.reload && !pw.isReloading && pw.ammo < pw.maxAmmo) {
    pw.isReloading = true; pw.reloadRemain = pw.reloadTime;
  }
  if (sim.interactivePlayer && playerActions.fireTarget && pw.ammo > 0 && pw.cdRemain <= 0 && !pw.isReloading) {
    pw.ammo--; pw.cdRemain = pw.fireCd;
    const toAgent: [number, number] = [ag.pos[0] - pl.pos[0], ag.pos[1] - pl.pos[1]];
    if (Math.hypot(toAgent[0], toAgent[1]) > 1) pl.facing = norm(toAgent);
    spawnProjectile(sim, pl.pos, ag.pos,
      { projSpeed: pw.projSpeed, baseDmg: pw.baseDmg, range: pw.range, canArc: false },
      false, pl.atkStat, pl.critChance, pl.critMult, playerTarget?.id ?? 0);
    const pr = sim.projectiles[sim.projectiles.length - 1];
    if (pr) pr.isPlayer = true;
  }

  // Scripted party members use the same role-aware movement/attack ordering
  // as CombatEnv.Target.tick_ai + _target_attacks_agent.
  for (const t of sim.targets) {
    if (!t.alive || (sim.interactivePlayer && t.isPlayerControlled)) continue;
    tickTargetAI(sim, t, dt);
  }

  // Projectiles use swept segment collision. Fixed world-space hit radii avoid
  // the old render-resolution-dependent physics.
  const projSubsteps = Math.max(1, Math.ceil(dt / (1 / 60)));
  const projDt = dt / projSubsteps;
  for (let sub = 0; sub < projSubsteps; sub++) {
    const active: Projectile[] = [];
    for (const p of sim.projectiles) {
      const oldX = p.pos[0], oldY = p.pos[1];
      if (sub === 0) p.life -= dt;

      if (p.canArc && p.arcStart && p.arcApex && p.arcEnd && p.arcFlightTime != null) {
        p.arcElapsed = (p.arcElapsed || 0) + projDt;
        const u = clamp(p.arcElapsed / Math.max(p.arcFlightTime, 0.01), 0, 1);
        const om = 1 - u;
        p.pos = [
          om * om * p.arcStart[0] + 2 * om * u * p.arcApex[0] + u * u * p.arcEnd[0],
          om * om * p.arcStart[1] + 2 * om * u * p.arcApex[1] + u * u * p.arcEnd[1],
        ];
        if (u >= 1) {
          if (p.isAgent) {
            for (const t of sim.targets) {
              if (!t.alive) continue;
              const dd = dist(p.pos, t.pos);
              if (dd < (p.arcImpactRadius || 150)) {
                const falloff = 1 - (dd / (p.arcImpactRadius || 150)) * 0.5;
                const hpBefore = t.hp;
                const result = computeDamage(p.damage * falloff, p.atkStat, t.defence, t.barrier, p.critChance, p.critMult, () => simRandom(sim));
                t.barrier = result.newBarrier; t.hp = Math.max(0, t.hp - result.hpDamage);
                if (t.hp <= 0) t.alive = false;
                recordAgentDamage(sim, t, hpBefore, result.hpDamage);
              }
            }
          } else if (!ag.isDodging && dist(p.pos, ag.pos) < (p.arcImpactRadius || 150)) {
            const dd = dist(p.pos, ag.pos), radius = p.arcImpactRadius || 150;
            const falloff = 1 - (dd / radius) * 0.5;
            const result = computeDamage(p.damage * falloff, p.atkStat, ag.defence, ag.barrier, p.critChance, p.critMult, () => simRandom(sim));
            ag.barrier = result.newBarrier; ag.hp = Math.max(0, ag.hp - result.hpDamage);
            recordAgentDamageTaken(sim, result.hpDamage);
          }
          continue;
        }
        if (p.life > 0) active.push(p);
        continue;
      }

      p.pos[0] += p.vel[0] * projDt; p.pos[1] += p.vel[1] * projDt;
      if (p.life <= 0 || Math.abs(p.pos[0]) > sim.half || Math.abs(p.pos[1]) > sim.half) continue;

      if (!p.canArc) {
        const dx = p.pos[0] - oldX, dy = p.pos[1] - oldY;
        if (sim.obstacles.some(o => rayAABB(oldX, oldY, dx, dy, o.x, o.y, o.hw, o.hh) !== null)) continue;
      }

      let hit = false;
      if (!p.isAgent) {
        if (!ag.isDodging && segPointDist(oldX, oldY, p.pos[0], p.pos[1], ag.pos[0], ag.pos[1]) < AGENT_BODY_RADIUS) {
          const result = computeDamage(p.damage, p.atkStat, ag.defence, ag.barrier, p.critChance, p.critMult, () => simRandom(sim));
          ag.barrier = result.newBarrier;
          ag.hp = Math.max(0, ag.hp - result.hpDamage);
          recordAgentDamageTaken(sim, result.hpDamage);
          const owner = p.ownerId ?? 0;
          ag.threatTable![owner] = (ag.threatTable![owner] || 0) + result.hpDamage;
          if (ag.hp <= 0) sim.done = true;
          hit = true;
        }
      } else {
        for (const t of sim.targets) {
          if (!t.alive) continue;
          if (segPointDist(oldX, oldY, p.pos[0], p.pos[1], t.pos[0], t.pos[1]) < 30) {
            const hpBefore = t.hp;
            const result = computeDamage(p.damage, p.atkStat, t.defence, t.barrier, p.critChance, p.critMult, () => simRandom(sim));
            t.barrier = result.newBarrier;
            t.hp = Math.max(0, t.hp - result.hpDamage);
            if (t.hp <= 0) t.alive = false;
            recordAgentDamage(sim, t, hpBefore, result.hpDamage);
            hit = true;
            break;
          }
        }
      }
      if (!hit) active.push(p);
    }
    sim.projectiles = active;
  }

  // Extension systems (statuses/allies) are finalized after base projectile
  // resolution and before reward/done/observation, matching Python exactly.
  tickExtendedSystems(sim, dt);

  if (playerTarget) {
    pl.hp = playerTarget.hp;
    if (pl.hp <= 0) pl.hp = 0;
  }
  sim.targetSnapshotElapsed += dt;
  if (!targetSnapshotIsValid(sim) || sim.targetSnapshotElapsed + 1e-9 >= 2.0) publishTargetSnapshot(sim);

  if (ag.hp <= 0 || sim.targets.every(t => !t.alive) || sim.step >= sim.maxSteps) sim.done = true;
  return sim;
}

function setLock(ag: AgentState, duration: number, reason: number) {
  ag.lockRemain = duration; ag.lockDuration = duration; ag.lockReason = reason;
}

function spawnProjectile(sim: SimState, from: [number, number], to: [number, number], slot: any, isAgent: boolean, atkStat = 0, critChance = 0, critMult = 1.5, ownerId?: number) {
  const d = dist(from, to);
  const dir = norm([to[0] - from[0], to[1] - from[1]]);
  const spread = (simRandom(sim) - 0.5) * 0.1; // Python uniform(-0.05, 0.05)
  const cs = Math.cos(spread), sn = Math.sin(spread);
  const fd: [number, number] = [dir[0] * cs - dir[1] * sn, dir[0] * sn + dir[1] * cs];
  const base: Projectile = {
    pos: [from[0], from[1]], vel: [fd[0] * slot.projSpeed, fd[1] * slot.projSpeed],
    damage: slot.baseDmg, atkStat: isAgent ? sim.agent.atkStat : (atkStat || 0),
    critChance: isAgent ? sim.agent.critChance : critChance,
    critMult: isAgent ? sim.agent.critMult : critMult,
    isAgent, canArc: slot.canArc || false,
    life: Math.max(d / Math.max(slot.projSpeed, 1) + 0.5, 1.0),
    ownerId,
  };
  if (base.canArc) {
    const midpoint: [number, number] = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2];
    const apex: [number, number] = [midpoint[0] + fd[0] * d * 0.1, midpoint[1] + fd[1] * d * 0.1];
    const arcLen = dist(from, apex) + dist(apex, to);
    base.arcStart = [...from]; base.arcApex = apex; base.arcEnd = [...to];
    base.arcElapsed = 0; base.arcFlightTime = arcLen / Math.max(slot.projSpeed, 500); base.arcImpactRadius = 150;
  }
  sim.projectiles.push(base);
}

// ═══════════════════════════════════════════════════════════════════
//  Score Target (Match Python _score_target exactly)
// ═══════════════════════════════════════════════════════════════════
function scoreTarget(sim: SimState, t: Target): number {
  const ag = sim.agent;
  const d = dist(ag.pos, t.pos);
  const normDist = Math.min(d / 2000, 1.0);
  let pcWeight = 30.0, lowHpWeight = 25.0, threatWeight = 20.0, distanceWeight = 10.0, losWeight = 15.0;
  if (sim.agentArchetype === 0) { losWeight *= 1.3; distanceWeight *= 0.8; }
  else if (sim.agentArchetype === 1) { distanceWeight *= 2.5; losWeight *= 0.5; lowHpWeight *= 1.5; pcWeight *= 1.2; }
  else if (sim.agentArchetype === 2) { threatWeight *= 0.3; pcWeight *= 0.5; distanceWeight *= 1.5; lowHpWeight *= 0.5; }
  else if (sim.agentArchetype === 3) { pcWeight *= 1.5; threatWeight *= 1.5; distanceWeight *= 1.3; lowHpWeight *= 0.5; }

  const pc = t.isPlayerControlled ? pcWeight : 0.0;
  const lowHp = (1.0 - t.hp / Math.max(t.maxHp, 1)) * lowHpWeight;
  let threat = 0.0;
  if (ag.threatTable) {
    const raw = ag.threatTable[t.id] || 0;
    const maxThreat = Math.max(0, ...Object.values(ag.threatTable));
    threat = maxThreat > 0 ? raw / maxThreat * threatWeight : 0;
  }
  const distanceScore = (1.0 - normDist) * distanceWeight;
  const los = checkLOS(ag.pos, t.pos, sim.obstacles) ? losWeight : 0.0;
  const current = currentTarget(sim);
  const sticky = current?.id === t.id ? 8.0 : 0.0;
  return pc + lowHp + threat + distanceScore + los + sticky;
}

// ═══════════════════════════════════════════════════════════════════
//  Calculate Frame Reward (Match Python Reward Function per Action)
// ═══════════════════════════════════════════════════════════════════

// Reward weights (synced with Python reward.py — last updated after phantom damage fix)
const RW = {
  damage_dealt: 0.15,              // Was 0.25 — reduced to prevent damage-only farming
  damage_taken: -0.015,
  kill_reward: 35.0,               // Was 20.0 — increased so focused kills dominate
  alive_per_step: -0.02,           // Survival COST — passive play is net-negative
  episode_win: 50.0,
  episode_loss: -5.0,
  episode_timeout: -8.0,           // Scales with episode length: -8 * (steps/200)
  surviving_target: -8.0,
  in_optimal_range: 0.01,
  range_closing: 0.04,
  out_of_range_penalty: -0.06,
  flanking_behind: 0.008,
  flanking_side: 0.003,
  damage_inactivity: -0.05,
  fire_in_optimal_band: 0.06,
  fire_outside_optimal: 0.0,
  reload_when_empty: 0.02,
  reload_behind_cover: 0.3,
  reload_in_open: -0.02,
  switch_to_loaded: 0.2,
  all_empty_penalty: -0.1,
  fire_hit: 0.15,
  invalid_action: -0.1,
  fire_valid_opportunity: 0.03,
  heavy_fire_intent_bonus: 0.02,
  wasted_shot: -0.0001,            // Was -0.03 — dramatically reduced
  passive_in_range: -0.08,         // NEW: per-step penalty for not firing when able
  target_low_hp_bonus: 3.0,        // NEW: one-time reward when target drops below 30%
  retarget_urgency: -0.06,         // NEW: per-step when selected target dead but others live
  arc_over_cover: 0.03,            // NEW: bonus for arcing over blocked LOS
  direct_fire_los: 0.02,           // NEW: bonus for direct weapon with clear LOS
};

interface RewardEngagementState {
  stepsSinceDamage: number;
  stepsSinceKill: number;
  lowHpSeen: Set<number>;
}

function calculateFrameReward(
  prev: SimState,
  curr: SimState,
  action: [number, number, number],
  stage: number,
  stepCount: number,
  dt: number,
  engState: RewardEngagementState,
): { reward: number; breakdown: any } {
  const agPrev = prev.agent;
  const agCurr = curr.agent;
  const [moveAction, combatAction, targetAction] = action;

  // ── 1. Engagement gate ───────────────────────────────────────
  // Python computes the gate from PRIOR history, then updates the counters
  // after this transition's damage has been attributed.
  const engagement = engState.stepsSinceDamage <= 4 ? 1.0 : 0.0;

  // ── 2. Authoritative agent damage / kills ────────────────────
  let totalDmgThisStep = 0;
  let totalDamageFractionThisStep = 0;
  for (const [idText, rawDamage] of Object.entries(curr.stepAgentDamageByTarget)) {
    const id = Number(idText);
    const target = curr.targets.find(t => t.id === id) || prev.targets.find(t => t.id === id);
    if (!target) continue;
    totalDmgThisStep += rawDamage;
    totalDamageFractionThisStep += rawDamage / Math.max(target.maxHp, 1);
  }
  const killsThisStep = curr.stepAgentKillIds.length;

  const activeTargetForReward = currentTarget(curr);
  let selectedDamageFraction = 0;
  if (activeTargetForReward) {
    const selectedRaw = curr.stepAgentDamageByTarget[activeTargetForReward.id] || 0;
    selectedDamageFraction = selectedRaw / Math.max(activeTargetForReward.maxHp, 1);
  }
  let focusMult = 1.0;
  if (totalDamageFractionThisStep > 1e-6 && activeTargetForReward) {
    const focusRatio = Math.min(selectedDamageFraction / totalDamageFractionThisStep, 1.0);
    focusMult = 0.5 + focusRatio;
  }
  const damageDealt = totalDamageFractionThisStep * RW.damage_dealt * 100 * focusMult;
  const killBonus = killsThisStep * RW.kill_reward;

  // ── 3. Damage taken ──────────────────────────────────────────
  const damageTakenFraction = curr.stepAgentDamageTaken / Math.max(agPrev.maxHp, 1);
  const damageTaken = damageTakenFraction * RW.damage_taken * 100;

  // ── 4. Time penalty ──────────────────────────────────────────
  const timePenalty = RW.alive_per_step;

  // ── 5. Range rewards (engagement-gated) ──────────────────────
  let optimalRange = 0;
  let rangeClosing = 0;
  let outOfRange = 0;

  const activeTarget = (agCurr.activeTargetIdx >= 0 && agCurr.activeTargetIdx < curr.targets.length
    ? curr.targets[agCurr.activeTargetIdx] : null) || curr.targets.find(t => t.alive);

  if (activeTarget && activeTarget.alive && stage >= 2) {
    const d = dist(agCurr.pos, activeTarget.pos);
    const slot = agCurr.weapons[agCurr.activeWeapon];
    const optMin = slot?.optMin || 0;
    const optMax = slot?.optMax || (slot?.range || 1500);
    const range = slot?.range || 1500;

    if (d >= optMin && d <= optMax) {
      optimalRange = RW.in_optimal_range * engagement;  // GATED
    }

    if (d > range) {
      const overshoot = d - range;
      const overshootFrac = overshoot / 800.0;
      outOfRange = RW.out_of_range_penalty * (0.3 + 0.7 * overshootFrac);

      // Range closing is only rewarded while outside ACTUAL weapon range.
      const prevTarget = prev.targets.find(t => t.id === activeTarget.id);
      if (prevTarget) {
        const prevD = dist(agPrev.pos, prevTarget.pos);
        const distClosed = prevD - d;
        if (distClosed > 0 && prevD > range) {
          const farBonus = 1.0 + Math.min(overshoot / 800.0, 2.0);
          rangeClosing = RW.range_closing * Math.min(distClosed / 80.0, 1.0) * farBonus;
        }
      }
    }
  }

  // ── 6. Flanking (engagement-gated, stage 3+) ────────────────
  let flanking = 0;
  if (activeTarget && activeTarget.alive && stage >= 3) {
    const toT = norm([activeTarget.pos[0] - agCurr.pos[0], activeTarget.pos[1] - agCurr.pos[1]]);
    const facingDot = dot(activeTarget.facing, toT);

    if (facingDot < -0.5) {
      flanking = RW.flanking_behind * engagement;  // GATED
    } else if (Math.abs(facingDot) < 0.5) {
      flanking = RW.flanking_side * engagement;    // GATED
    }
  }

  // ── 7. Inactivity penalty (stage 3+) ────────────────────────
  let inactivity = 0;
  // Update trackers AFTER computing this transition's engagement gate.
  if (totalDamageFractionThisStep > 0) engState.stepsSinceDamage = 0;
  else engState.stepsSinceDamage++;
  const anyKillThisStep = prev.targets.some(pt => pt.alive && curr.targets.some(ct => ct.id === pt.id && !ct.alive));
  if (anyKillThisStep) engState.stepsSinceKill = 0;
  else engState.stepsSinceKill++;

  if (stage >= 3) {
    let threshold = stage <= 4 ? 10 : (stage === 5 ? 13 : 10);
    if (engState.stepsSinceKill < 30) threshold = Math.max(threshold, 35);
    if (engState.stepsSinceDamage >= threshold) {
      const idleSteps = engState.stepsSinceDamage - threshold;
      const escalation = 1.0 + Math.min(idleSteps / 20.0, 2.0);
      inactivity = RW.damage_inactivity * escalation;
    }
  }

  // ── 8. Weapon selection (engagement-gated, stage 4+) ────────
  let weaponSelection = 0;
  if (stage >= 4 && combatAction === 1 && activeTarget?.alive) {
    const slot = agCurr.weapons[agCurr.activeWeapon];
    if (slot) {
      const d = dist(agCurr.pos, activeTarget.pos);
      const inBand = d >= (slot.optMin || 0) && d <= (slot.optMax || slot.range);
      if (inBand && totalDmgThisStep > 0) {
        weaponSelection = RW.fire_in_optimal_band * engagement;  // GATED + requires hit
      } else if (!inBand) {
        weaponSelection = RW.fire_outside_optimal * engagement;
      }
    }
  }

  // ── 9. Ammo management ──────────────────────────────────────
  let ammo = 0;
  const prevSlot = agPrev.weapons[agPrev.activeWeapon];
  const prevTarget = currentTarget(prev);
  const prevTargetDistance = prevTarget?.alive ? dist(agPrev.pos, prevTarget.pos) : 9999;
  const prevHasLOS = !!prevTarget?.alive && checkLOS(agPrev.pos, prevTarget.pos, prev.obstacles);
  let prevBlockedHeight = 0;
  if (prevTarget?.alive && !prevHasLOS) {
    const dx = prevTarget.pos[0] - agPrev.pos[0], dy = prevTarget.pos[1] - agPrev.pos[1];
    for (const o of prev.obstacles) {
      if (rayAABB(agPrev.pos[0], agPrev.pos[1], dx, dy, o.x, o.y, o.hw, o.hh) !== null) { prevBlockedHeight = o.height; break; }
    }
  }
  const prevCanArcCover = !!prevSlot?.canArc && !prevHasLOS && (prevSlot.maxArcHeight == null || prevSlot.maxArcHeight <= 0 || prevBlockedHeight <= prevSlot.maxArcHeight);
  const prevCanFire = !!prevSlot && prevSlot.cdRemain <= 0 && prevSlot.ammo > 0 && !prevSlot.isReloading && !agPrev.isSwitching && !agPrev.isDodging && !agPrev.isWindingUp;
  const validFireOpportunity = !!prevTarget?.alive && prevCanFire && prevTargetDistance <= (prevSlot?.range || 0) && (prevHasLOS || prevCanArcCover);

  if (stage >= 2 && combatAction === 2 && prevSlot) {
    const currTargetForCover = currentTarget(curr);
    const behindCover = !!currTargetForCover?.alive && !checkLOS(agCurr.pos, currTargetForCover.pos, curr.obstacles);
    ammo += behindCover ? RW.reload_behind_cover : RW.reload_in_open;
  }
  if (stage >= 2 && (combatAction === 3 || combatAction === 4)) {
    const currSlot = agCurr.weapons[agCurr.activeWeapon];
    if (prevSlot && currSlot && prevSlot.ammo <= 0 && currSlot.ammo > 0) ammo += RW.switch_to_loaded;
  }
  if (stage >= 2 && combatAction === 1 && prevCanFire) {
    if (totalDamageFractionThisStep > 0) ammo += RW.fire_hit;
    else if (!prevTarget?.alive || prevTargetDistance > (prevSlot?.range || 0) || (!prevHasLOS && !prevCanArcCover)) ammo += RW.wasted_shot;
  }
  const prevAllEmpty = agPrev.weapons.every(w => w.ammo <= 0);
  const currAllEmpty = agCurr.weapons.every(w => w.ammo <= 0);
  if (stage >= 2 && currAllEmpty && !prevAllEmpty) ammo += RW.all_empty_penalty;

  // Invalid-action and immediate fire-intent/passivity terms.
  let invalidAction = 0;
  if (combatAction === 1 && !prevCanFire) invalidAction = RW.invalid_action;
  else if (combatAction === 2 && prevSlot && prevSlot.ammo >= prevSlot.maxAmmo) invalidAction = RW.invalid_action;
  let fireIntent = 0;
  if (stage >= 2 && combatAction === 1 && validFireOpportunity) {
    fireIntent = RW.fire_valid_opportunity;
    if ((prevSlot?.windUp || 0) >= 0.3 || (prevSlot?.fireCd || 0) >= 0.8) fireIntent += RW.heavy_fire_intent_bonus;
  }
  let passive = 0;
  const usefulBlock = combatAction === 6 && curr.stepAgentDamageTaken > 0;
  if ((combatAction === 0 || combatAction === 6) && !usefulBlock && validFireOpportunity) passive = RW.passive_in_range;

  // ── 10. Multi-target progression ────────────────────────────
  let retargetUrgency = 0;
  let targetLowHp = 0;
  if (activeTargetForReward && !activeTargetForReward.alive && curr.targets.some(t => t.alive)) retargetUrgency = RW.retarget_urgency;
  if (activeTargetForReward && activeTargetForReward.alive && activeTargetForReward.hp > 0 && activeTargetForReward.hp / Math.max(activeTargetForReward.maxHp, 1) < 0.3
      && selectedDamageFraction > 0 && !engState.lowHpSeen.has(activeTargetForReward.id)) {
    targetLowHp = RW.target_low_hp_bonus;
    engState.lowHpSeen.add(activeTargetForReward.id);
  }

  // ── 11. Episode end ─────────────────────────────────────────
  let endBonus = 0;
  if (curr.done && !prev.done) {
    if (agCurr.hp <= 0) {
      endBonus = -10.0 + RW.episode_loss;
    } else {
      const aliveHostiles = curr.targets.filter(t => t.alive).length;
      if (aliveHostiles === 0) {
        const speedBonus = 1.0 + 0.5 * Math.max(0, 1.0 - stepCount / 500.0);
        endBonus = RW.episode_win * speedBonus;
      } else {
        // Timeout penalty scales with episode length
        const lengthScale = Math.max(1.0, stepCount / 200.0);
        endBonus = RW.episode_timeout * lengthScale + aliveHostiles * RW.surviving_target;
      }
    }
  }

  // ── Total ───────────────────────────────────────────────────
  const total = damageDealt + killBonus + damageTaken + timePenalty +
    optimalRange + rangeClosing + outOfRange + retargetUrgency + targetLowHp +
    flanking + inactivity + weaponSelection + ammo + invalidAction + fireIntent + passive + endBonus;

  return {
    reward: total,
    breakdown: {
      damageDealt,
      killBonus,
      damageTaken,
      timePenalty,
      optimalRange,
      rangeClosing,
      outOfRange,
      flanking,
      inactivity,
      weaponSelection,
      ammo: ammo + invalidAction + fireIntent + passive + retargetUrgency + targetLowHp,
      endBonus,
      engagement,  // Show this in the chart so you can see the gate
    },
  };
}

// ═══════════════════════════════════════════════════════════════════
//  Build 249-float Observation
// ═══════════════════════════════════════════════════════════════════
const TIER_PROFILES = {
  micro: { name: "Micro", frameStack: 3, decisionInterval: DT, spatialTraces: 8, sizeLabel: "Combat_Micro.onnx" },
  small: { name: "Small", frameStack: 3, decisionInterval: DT, spatialTraces: 8, sizeLabel: "Combat_Small.onnx" },
  medium: { name: "Medium", frameStack: 3, decisionInterval: DT, spatialTraces: 8, sizeLabel: "Combat_Medium.onnx" },
  large: { name: "Large", frameStack: 3, decisionInterval: DT, spatialTraces: 8, sizeLabel: "Combat_Large.onnx" },
};

function softmaxSample(logits: number[], mask: boolean[], temp = 1.0): number {
  let maxLogit = -1e30;
  const maskedLogits = logits.map((val, idx) => {
    if (!mask[idx]) return -1e30;
    if (val > maxLogit) maxLogit = val;
    return val;
  });

  const exps = maskedLogits.map((val, idx) => {
    if (!mask[idx]) return 0;
    return Math.exp((val - maxLogit) / Math.max(temp, 0.05));
  });

  const sumExps = exps.reduce((a, b) => a + b, 0);
  if (sumExps <= 0) {
    const defaultIdx = mask.indexOf(true);
    return defaultIdx !== -1 ? defaultIdx : 0;
  }

  const probs = exps.map(v => v / sumExps);
  let roll = Math.random();
  for (let i = 0; i < probs.length; i++) {
    roll -= probs[i];
    if (roll <= 0) return i;
  }
  return probs.length - 1;
}

// ═══════════════════════════════════════════════════════════════════
//  Build 249-float Observation
// ═══════════════════════════════════════════════════════════════════
function buildObservation(
  sim: SimState,
  playerPos: [number, number],
  prevTargetVelMap: Record<number, [number, number]> = {},
  decisionInterval = 0.2
): Float32Array {
  const obs = new Float32Array(OBS_SIZE);
  const ag = sim.agent;
  const slot = ag.weapons[ag.activeWeapon] || null;
  const nSlots = ag.weapons.length;
  let idx = 0;

  obs[idx++] = ag.hp / ag.maxHp;
  obs[idx++] = clamp(ag.defence / 200, 0, 1);
  const speed = Math.sqrt(ag.vel[0] ** 2 + ag.vel[1] ** 2);
  obs[idx++] = ag.maxSpeed > 0 ? clamp(speed / ag.maxSpeed, 0, 1) : 0;
  obs[idx++] = sim.status.stunRemain > 0 ? 1 : 0;
  obs[idx++] = sim.status.slowRemain > 0 ? 1 : 0;
  for (let i = 0; i < 6; i++) obs[idx++] = sim.status.debuffs[i]?.remain > 0 ? sim.status.debuffs[i].strength : 0;
  const velDir = speed > 1 ? [ag.vel[0] / speed, ag.vel[1] / speed] : [0, 0];
  obs[idx++] = velDir[0];
  obs[idx++] = velDir[1];
  obs[idx++] = clamp(ag.combatTime / 120, 0, 1);

  obs[idx++] = 0.176; // grounded capsule trace, matches Python extended env

  obs[idx++] = ag.lockRemain > 0 ? 1 : 0;
  const lockProg = ag.lockRemain > 0 ? clamp(1 - ag.lockRemain / Math.max(ag.lockDuration, 0.01), 0, 1) : 0;
  obs[idx++] = lockProg;
  obs[idx++] = ag.lockRemain > 0 ? ag.lockReason / 7.0 : 0;
  obs[idx++] = ag.isDodging ? 1 : 0;
  obs[idx++] = ag.dodgeCd <= 0 ? 1 : 0;
  obs[idx++] = ag.isDodging ? 1 : 0;

  obs[idx++] = nSlots > 1 ? ag.activeWeapon / (nSlots - 1) : 0;
  obs[idx++] = slot ? slot.ammo / slot.maxAmmo : 0;
  obs[idx++] = (slot && slot.cdRemain <= 0 && slot.ammo > 0 && !slot.isReloading) ? 1 : 0;
  obs[idx++] = (slot && slot.isReloading) ? 1 : 0;
  obs[idx++] = (slot && slot.isReloading && slot.reloadTime > 0)
    ? clamp(1 - slot.reloadRemain / slot.reloadTime, 0, 1) : 1.0;
  obs[idx++] = slot ? clamp(slot.range / 5000, 0, 1) : 0;
  obs[idx++] = (slot && slot.fireCd > 0) ? clamp(slot.cdRemain / slot.fireCd, 0, 1) : 0;
  obs[idx++] = slot ? clamp((slot.windUp || 0) / 3.0, 0, 1) : 0;
  obs[idx++] = (slot && slot.canArc) ? 1 : 0;
  obs[idx++] = 1; // Is ranged

  for (let si = 0; si < 3; si++) {
    const actual = si < ag.activeWeapon ? si : si + 1;
    if (actual < nSlots) {
      const w = ag.weapons[actual];
      obs[idx++] = w.ammo / w.maxAmmo;
      obs[idx++] = clamp(w.range / 5000, 0, 1);
      obs[idx++] = w.isReloading ? 1 : 0;
      obs[idx++] = w.canArc ? 1 : 0;
    } else {
      idx += 4;
    }
  }

  for (let ai = 0; ai < 4; ai++) obs[idx++] = sim.agentArchetype === ai ? 1 : 0;
  const weaponRange = slot ? slot.range : 1000;
  obs[idx++] = clamp(weaponRange * 0.6 / 5000, 0, 1);
  const anyAmmo = ag.weapons.some(w => w.ammo > 0);
  obs[idx++] = anyAmmo ? 1 : 0;
  obs[idx++] = ag.melee.cdRemain <= 0 ? 1 : 0;

  // Hostile slots must use the same immutable published priority snapshot as
  // target-action masking/selection, not a fresh sort at observation time.
  const sortedTargets = getSortedTargets(sim);
  const ct = currentTarget(sim);

  if (ct && ct.alive) {
    const rel = [ct.pos[0] - ag.pos[0], ct.pos[1] - ag.pos[1]];
    const d = Math.sqrt(rel[0] ** 2 + rel[1] ** 2) || 1;
    const hasLOS = checkLOS(ag.pos, ct.pos, sim.obstacles);
    obs[idx++] = clamp(rel[0] / 5000, -1, 1);
    obs[idx++] = clamp(rel[1] / 5000, -1, 1);
    obs[idx++] = clamp(d / 5000, 0, 1);
    obs[idx++] = ct.hp / ct.maxHp;
    obs[idx++] = d <= weaponRange ? 1 : 0;
    obs[idx++] = hasLOS ? 1 : 0;
    const facingToTarget = dot(ag.facing, [rel[0] / d, rel[1] / d]);
    obs[idx++] = facingToTarget > -0.17 ? 1 : 0;
    obs[idx++] = dot(ag.facing, [rel[0] / d, rel[1] / d]);
    const toAgDir = [ag.pos[0] - ct.pos[0], ag.pos[1] - ct.pos[1]];
    const toAgD = Math.sqrt(toAgDir[0] ** 2 + toAgDir[1] ** 2) || 1;
    obs[idx++] = dot(ct.facing, [toAgDir[0] / toAgD, toAgDir[1] / toAgD]); // [-1,1] raw dot, matches C++
    obs[idx++] = clamp(ct.vel[0] / 600, -1, 1);
    obs[idx++] = clamp(ct.vel[1] / 600, -1, 1);

    // Compute exact target acceleration (PrimaryTarget indices 11, 12 - C++ alignment)
    const prevVel = prevTargetVelMap[ct.id] || [0, 0];
    const currVel = ct.vel || [0, 0];
    const accelX = (currVel[0] - prevVel[0]) / Math.max(decisionInterval, 0.01);
    const accelY = (currVel[1] - prevVel[1]) / Math.max(decisionInterval, 0.01);
    obs[idx++] = clamp(accelX / 2000, -1, 1);
    obs[idx++] = clamp(accelY / 2000, -1, 1);

    obs[idx++] = clamp(70.0 / Math.max(d, 1), 0, 1);
    obs[idx++] = ct.isPlayerControlled ? 1 : 0; // is_player_controlled

    let behindCover = false, coverH = 0;
    for (const o of sim.obstacles) {
      const dx = ct.pos[0] - ag.pos[0], dy = ct.pos[1] - ag.pos[1];
      if (rayAABB(ag.pos[0], ag.pos[1], dx, dy, o.x, o.y, o.hw, o.hh) !== null) {
        behindCover = true; coverH = o.height; break;
      }
    }
    obs[idx++] = (behindCover && coverH < 200) ? 1 : 0;
    obs[idx++] = behindCover ? clamp(coverH / 500, 0, 1) : 0;
    obs[idx++] = d <= (ag.melee.range || 200) ? 1 : 0;
    const tVel = ct.vel || [0, 0];
    const relVel: [number, number] = [ag.vel[0] - tVel[0], ag.vel[1] - tVel[1]];
    const toTarget: [number, number] = [rel[0] / d, rel[1] / d];
    obs[idx++] = clamp(dot(relVel, toTarget) / 1000, -1, 1);
    obs[idx++] = ct.characterType;
    obs[idx++] = ct.maxMana > 0 ? clamp(ct.mana / ct.maxMana, 0, 1) : 0;
    obs[idx++] = clamp(ct.commitment, 0, 1);
    obs[idx++] = gapCloserThreat(ct, ag.pos);
    obs[idx++] = (!ag.isRepositioning && ag.repositionCd <= 0) ? 1 : 0;
  } else {
    // Reposition readiness remains self-owned even without a live target.
    idx += 23;
    obs[idx++] = (!ag.isRepositioning && ag.repositionCd <= 0) ? 1 : 0;
  }

  for (let si = 0; si < 4; si++) {
    const base = 74 + si * 17;
    if (si < sortedTargets.length) {
      const t = sortedTargets[si];
      const rel = [t.pos[0] - ag.pos[0], t.pos[1] - ag.pos[1]];
      const d = Math.sqrt(rel[0] ** 2 + rel[1] ** 2) || 1;
      obs[base] = 1;
      obs[base + 1] = clamp(rel[0] / 5000, -1, 1);
      obs[base + 2] = clamp(rel[1] / 5000, -1, 1);
      obs[base + 3] = clamp(d / 5000, 0, 1);
      obs[base + 4] = sim.targetSlotHps[t.id] ?? (t.hp / t.maxHp);
      obs[base + 5] = (sim.targetSlotLos[t.id] ?? checkLOS(ag.pos, t.pos, sim.obstacles)) ? 1 : 0;
      obs[base + 6] = t.isPlayerControlled ? 1 : 0; // is_player_controlled
      const toAg = norm([ag.pos[0] - t.pos[0], ag.pos[1] - t.pos[1]]);
      obs[base + 7] = clamp(dot(t.facing, toAg), -1, 1); // raw dot [-1,1], matches C++
      obs[base + 8] = clamp((sim.targetSlotScores[t.id] ?? scoreTarget(sim, t)) / 120.0, 0, 1); // published score
      const targetThreat = ag.threatTable ? (ag.threatTable[t.id] || 0) : 0;
      obs[base + 9] = clamp(targetThreat / 200.0, 0, 1);
      obs[base + 10] = clamp(t.vel[0] / 600, -1, 1);
      obs[base + 11] = clamp(t.vel[1] / 600, -1, 1);
      obs[base + 12] = clamp(dot(t.facing, toAg), 0, 1);
      obs[base + 13] = t.characterType;
      obs[base + 14] = t.maxMana > 0 ? clamp(t.mana / t.maxMana, 0, 1) : 0;
      obs[base + 15] = clamp(t.commitment, 0, 1);
      obs[base + 16] = gapCloserThreat(t, ag.pos);
    }
  }
  idx = 142;
  for (let si = 0; si < 3; si++) {
    const ally = sim.allies[si];
    if (!ally || !ally.alive) { idx += 15; continue; }
    const rel: [number, number] = [ally.pos[0] - ag.pos[0], ally.pos[1] - ag.pos[1]];
    const ad = Math.hypot(rel[0], rel[1]) || 1;
    obs[idx++] = 1;
    obs[idx++] = clamp(rel[0] / 5000, -1, 1);
    obs[idx++] = clamp(rel[1] / 5000, -1, 1);
    obs[idx++] = clamp(ad / 5000, 0, 1);
    obs[idx++] = clamp(ally.hp / Math.max(ally.maxHp, 1), 0, 1);
    obs[idx++] = checkLOS(ag.pos, ally.pos, sim.obstacles) ? 1 : 0;
    obs[idx++] = clamp(ally.vel[0] / 600, -1, 1);
    obs[idx++] = clamp(ally.vel[1] / 600, -1, 1);
    const toMe = norm([ag.pos[0] - ally.pos[0], ag.pos[1] - ally.pos[1]]);
    obs[idx++] = clamp(dot(ally.facing, toMe), -1, 1);
    obs[idx++] = ally.attackCooldown > 0 ? clamp(1 - ally.attackCd / ally.attackCooldown, 0, 1) : 1;
    obs[idx++] = 0; // scripted allies do not reload
    obs[idx++] = clamp(ally.attackCd / 2, 0, 1);
    const allyTgt = sortedTargets.findIndex(t => t.id === ally.targetId);
    obs[idx++] = (allyTgt + 1) / 5;
    obs[idx++] = clamp(ally.combatAction / 8, 0, 1);
    let flank = 0;
    if (allyTgt >= 0) {
      const shared = sortedTargets[allyTgt];
      const myTo = norm([shared.pos[0] - ag.pos[0], shared.pos[1] - ag.pos[1]]);
      const allyTo = norm([shared.pos[0] - ally.pos[0], shared.pos[1] - ally.pos[1]]);
      flank = dot(myTo, allyTo);
    }
    obs[idx++] = flank;
  }

  const spatialTraceLen = 1500;
  const effectiveHalf = sim.half - AGENT_BODY_RADIUS;
  for (const ang of SPATIAL_ANGLES) {
    const rad = ang * Math.PI / 180;
    const dx = Math.cos(rad) * spatialTraceLen, dy = Math.sin(rad) * spatialTraceLen;
    let minT = 1.0;
    // Sphere sweep is equivalent to a point ray against each AABB inflated
    // by the capsule/body radius.
    for (const o of sim.obstacles) {
      const t = rayAABB(ag.pos[0], ag.pos[1], dx, dy, o.x, o.y, o.hw + AGENT_BODY_RADIUS, o.hh + AGENT_BODY_RADIUS);
      if (t !== null && t < minT) minT = Math.max(0, t);
    }
    const deltas = [dx, dy];
    const positions = [ag.pos[0], ag.pos[1]];
    for (let axis = 0; axis < 2; axis++) {
      const dAxis = deltas[axis];
      if (Math.abs(dAxis) <= 1e-6) continue;
      const wall = dAxis > 0 ? effectiveHalf : -effectiveHalf;
      const tWall = (wall - positions[axis]) / dAxis;
      if (tWall >= 0 && tWall < minT) minT = tWall;
    }
    obs[idx++] = clamp(minT, 0, 1);
  }

  // ── Cover Height (8) — continuous obstacle height / 500 ──
  const HIGH_TRACE_HEIGHT = 350;
  for (const ang of SPATIAL_ANGLES) {
    const rad = ang * Math.PI / 180;
    const dx = Math.cos(rad) * 1500, dy = Math.sin(rad) * 1500;
    let obsHeight = 0;
    for (const o of sim.obstacles) {
      if (rayAABB(ag.pos[0], ag.pos[1], dx, dy, o.x, o.y, o.hw, o.hh) !== null) {
        obsHeight = o.height < HIGH_TRACE_HEIGHT ? o.height : HIGH_TRACE_HEIGHT;
        break;
      }
    }
    obs[idx++] = clamp(obsHeight / 500, 0, 1);
  }

  let nearProjDist = 1, nearProjTTA = 1, nearProjDir = [0, 0];
  for (const p of sim.projectiles) {
    if (p.isAgent) continue;
    const d = dist(p.pos, ag.pos);
    if (d > 600) continue;
    const spd = Math.sqrt(p.vel[0] ** 2 + p.vel[1] ** 2); if (spd < 1) continue;
    const vd: [number, number] = [p.vel[0] / spd, p.vel[1] / spd];
    const toMe = norm([ag.pos[0] - p.pos[0], ag.pos[1] - p.pos[1]]);
    if (dot(vd, toMe) <= 0.5) continue;
    const nd = d / 600;
    if (nd < nearProjDist) { nearProjDist = nd; nearProjDir = vd; nearProjTTA = clamp(d / spd, 0, 2) / 2; }
  }
  obs[idx++] = nearProjDist;
  obs[idx++] = nearProjTTA;
  obs[idx++] = nearProjDir[0];
  obs[idx++] = nearProjDir[1];

  let nearMeleeDist = 1, nearMeleeDir = [0, 0];
  for (const t of sim.targets) {
    if (!t.alive) continue;
    const d = dist(ag.pos, t.pos);
    if (d > 350) continue;
    const nd = d / 350;
    if (nd < nearMeleeDist) { nearMeleeDist = nd; nearMeleeDir = norm([t.pos[0] - ag.pos[0], t.pos[1] - ag.pos[1]]); }
  }
  obs[idx++] = nearMeleeDist;
  obs[idx++] = nearMeleeDir[0];
  obs[idx++] = nearMeleeDir[1];
  obs[idx++] = ag.dodgeCd <= 0 ? 1 : 0;

  const navDirs = [[1, 0], [S, S], [0, 1], [-S, S], [-1, 0], [-S, -S], [0, -1], [S, -S], [0, 0]];
  for (const nd of navDirs) {
    const testX = ag.pos[0] + nd[0] * 400, testY = ag.pos[1] + nd[1] * 400;
    let blocked = Math.abs(testX) > sim.half || Math.abs(testY) > sim.half;
    if (!blocked) {
      for (const o of sim.obstacles) {
        if (Math.abs(testX - o.x) < o.hw && Math.abs(testY - o.y) < o.hh) { blocked = true; break; }
      }
    }
    obs[idx++] = blocked ? 0 : 1;
  }

  const aliveH = sim.targets.filter(t => t.alive).length;
  const liveAllies = sim.allies.filter(a => a.alive);
  const aliveA = liveAllies.length;
  obs[idx++] = clamp(aliveA / 10, 0, 1);
  obs[idx++] = clamp(aliveH / 4, 0, 1);
  obs[idx++] = aliveA > 0 ? liveAllies.reduce((s, a) => s + a.hp / Math.max(a.maxHp, 1), 0) / aliveA : 0;
  obs[idx++] = aliveH > 0 ? sim.targets.filter(t => t.alive).reduce((s, t) => s + t.hp / t.maxHp, 0) / aliveH : 0;
  const groupTotal = aliveA + aliveH;
  obs[idx++] = groupTotal > 0 ? aliveA / groupTotal : 0.5;
  obs[idx++] = aliveH > aliveA ? 1 : 0;

  obs[idx++] = clamp(dist(ag.pos, ag.spawnPos || [0, 0]) / (ag.leashRange || 2000), 0, 1);

  // ── Extended Threat: Projectiles 2 and 3 (227-232) ──
  // Collect up to 3 nearest threatening projectiles
  const threatProjs: { dist: number; dir: [number, number] }[] = [];
  for (const p of sim.projectiles) {
    if (p.isAgent) continue;
    const d = dist(p.pos, ag.pos);
    if (d > 600) continue;
    const spd = Math.sqrt(p.vel[0] ** 2 + p.vel[1] ** 2); if (spd < 1) continue;
    const vd: [number, number] = [p.vel[0] / spd, p.vel[1] / spd];
    const toMe = norm([ag.pos[0] - p.pos[0], ag.pos[1] - p.pos[1]]);
    if (dot(vd, toMe) <= 0.5) continue;
    threatProjs.push({ dist: d / 600, dir: vd });
  }
  threatProjs.sort((a, b) => a.dist - b.dist);
  // Proj 2 (indices 227-229)
  if (threatProjs.length > 1) {
    obs[idx++] = threatProjs[1].dist;
    obs[idx++] = threatProjs[1].dir[0];
    obs[idx++] = threatProjs[1].dir[1];
  } else { obs[idx++] = 1; obs[idx++] = 0; obs[idx++] = 0; }
  // Proj 3 (indices 230-232)
  if (threatProjs.length > 2) {
    obs[idx++] = threatProjs[2].dist;
    obs[idx++] = threatProjs[2].dir[0];
    obs[idx++] = threatProjs[2].dir[1];
  } else { obs[idx++] = 1; obs[idx++] = 0; obs[idx++] = 0; }
  // Threat count (index 233)
  obs[idx++] = clamp(threatProjs.length / 5, 0, 1);

  // ── Can Hit Target per Weapon (234-237) ──
  const tgt = currentTarget(sim);
  const tgtDist = tgt && tgt.alive ? dist(ag.pos, tgt.pos) : 9999;
  const tgtLOS = tgt && tgt.alive ? checkLOS(ag.pos, tgt.pos, sim.obstacles) : false;
  let tgtBehindLowCover = false;
  if (tgt && tgt.alive && !tgtLOS) {
    for (const o of sim.obstacles) {
      const dx = tgt.pos[0] - ag.pos[0], dy = tgt.pos[1] - ag.pos[1];
      if (rayAABB(ag.pos[0], ag.pos[1], dx, dy, o.x, o.y, o.hw, o.hh) !== null && o.height < 350) {
        tgtBehindLowCover = true; break;
      }
    }
  }
  for (let wi = 0; wi < 4; wi++) {
    if (wi < ag.weapons.length) {
      const w = ag.weapons[wi];
      const inRange = tgtDist <= w.range;
      const hasPath = tgtLOS || (w.canArc && tgtBehindLowCover);
      const canHit = w.ammo > 0 && inRange && hasPath && !w.isReloading;
      obs[idx++] = canHit ? 1 : 0;
    } else { idx++; }
  }

  // ── Total Ammo Fraction (238) ──
  const totalAmmo = ag.weapons.length > 0
    ? ag.weapons.reduce((s, w) => s + w.ammo / w.maxAmmo, 0) / ag.weapons.length
    : 0;
  obs[idx++] = clamp(totalAmmo, 0, 1);

  // ── Targets Killed Fraction (239) ──
  const totalHostiles = sim.targets.length;
  const killedHostiles = sim.targets.filter(t => !t.alive).length;
  obs[idx++] = totalHostiles > 0 ? killedHostiles / totalHostiles : 0;

  // ── Arc Clearance per Weapon (240-243) ──
  for (let wi = 0; wi < 4; wi++) {
    if (wi < ag.weapons.length) {
      const w = ag.weapons[wi];
      if (w.canArc) {
        const maxArc = w.maxArcHeight || 0;
        obs[idx++] = maxArc <= 0 ? 1.0 : clamp(maxArc / 3000, 0, 1);
      } else { idx++; }
    } else { idx++; }
  }

  // ── Player Patterns (244-248) ──
  obs[idx++] = clamp(sim.playerPatterns.aggression, 0, 1);
  obs[idx++] = clamp(sim.playerPatterns.evasion, 0, 1);
  obs[idx++] = clamp(sim.playerPatterns.predictability, 0, 1);
  obs[idx++] = clamp(sim.playerPatterns.preferredRange, 0, 1);
  obs[idx++] = clamp(sim.playerPatterns.manaBurn, 0, 1);

  if (idx !== OBS_SIZE) {
    throw new Error(`Observation builder wrote ${idx} fields; expected ${OBS_SIZE}.`);
  }
  return obs;
}

// ═══════════════════════════════════════════════════════════════════
//  Build Action Mask
// ═══════════════════════════════════════════════════════════════════
function buildActionMask(sim: SimState) {
  const ag = sim.agent;
  const slot = ag.weapons[ag.activeWeapon];
  const isLocked = ag.lockRemain > 0;
  const m = new Array(MOVEMENT_ACTIONS).fill(true);
  const c = new Array(COMBAT_ACTIONS).fill(false);
  const t = new Array(TARGET_ACTIONS).fill(false);
  c[0] = true;

  if (isLocked) {
    m.fill(false);
    m[clamp(ag.cachedMovementAction, 0, MOVEMENT_ACTIONS - 1)] = true;
    t[TARGET_ACTIONS - 1] = true;
    return { m, c, t, skipInference: true };
  }

  // Extended-env stun overrides the executable action but does NOT freeze
  // recurrent inference. Python only freezes inference for explicit locks.
  if (sim.status.stunRemain > 0) {
    m.fill(false); m[0] = true;
    c.fill(false); c[0] = true;
    t.fill(false); t[TARGET_ACTIONS - 1] = true;
    return { m, c, t, skipInference: false };
  }

  if (ag.hp > 0 && !ag.isSwitching) {
    if (slot && slot.cdRemain <= 0 && slot.ammo > 0 && !slot.isReloading && !ag.isWindingUp && !ag.isDodging) c[1] = true;
    if (slot && slot.maxAmmo > 0 && slot.ammo < slot.maxAmmo && !slot.isReloading) c[2] = true;
    if (ag.weapons.length > 0 && ag.activeWeapon !== 0) c[3] = true;
    if (ag.weapons.length > 1 && ag.activeWeapon !== 1) c[4] = true;
    if (ag.melee.cdRemain <= 0 && !ag.isDodging) c[5] = true;
    c[6] = true;
    if (!ag.isDodging && ag.dodgeCd <= 0) c[7] = true;
    if (!ag.isRepositioning && ag.repositionCd <= 0) c[8] = true;
  }

  const sorted = getSortedTargets(sim);
  for (let i = 0; i < TARGET_ACTIONS - 1; i++) t[i] = i < sorted.length && sorted[i].alive;
  t[TARGET_ACTIONS - 1] = true;
  return { m, c, t, skipInference: false };
}

// ═══════════════════════════════════════════════════════════════════
//  ONNX Inference
// ═══════════════════════════════════════════════════════════════════
interface LoadedModel {
  session: any;
  frameStack: number;
  hiddenSize: number;
  hidden: Float32Array;
}

async function loadOnnxModel(file: File) {
  const buf = await file.arrayBuffer();
  const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  const requiredInputs = ["observation", "hidden_in"];
  const requiredOutputs = ["movement_logits", "combat_logits", "target_logits", "hidden_out"];

  for (const name of requiredInputs) {
    if (!session.inputNames.includes(name)) {
      await session.release();
      throw new Error(`Unsupported model: missing input '${name}'.`);
    }
  }
  for (const name of requiredOutputs) {
    if (!session.outputNames.includes(name)) {
      await session.release();
      throw new Error(`Unsupported model: missing output '${name}'.`);
    }
  }

  const obsMeta: any = session.inputMetadata[session.inputNames.indexOf("observation")];
  const hiddenMeta: any = session.inputMetadata[session.inputNames.indexOf("hidden_in")];
  const expectedInput = OBS_SIZE * FRAME_STACK;
  const obsWidth = obsMeta?.isTensor ? obsMeta.shape?.[obsMeta.shape.length - 1] : undefined;
  const hiddenSize = hiddenMeta?.isTensor ? hiddenMeta.shape?.[hiddenMeta.shape.length - 1] : undefined;
  if (obsWidth !== expectedInput) {
    await session.release();
    throw new Error(`Unsupported model observation width ${String(obsWidth)}; expected ${expectedInput}.`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    await session.release();
    throw new Error("Unsupported model: hidden_in must have shape [1, batch, hidden_size].");
  }

  const model: LoadedModel = {
    session,
    frameStack: FRAME_STACK,
    hiddenSize,
    hidden: new Float32Array(hiddenSize),
  };

  // A real dry run validates all independent-head widths before accepting the
  // upload; incompatible legacy models fail here instead of sampling nonsense.
  try {
    await runInference(model, new Float32Array(expectedInput), model.hidden);
    return model;
  } catch (error) {
    await session.release();
    throw error;
  }
}

async function runInference(model: LoadedModel, obsBuffer: Float32Array, hidden = model.hidden) {
  if (obsBuffer.length !== OBS_SIZE * FRAME_STACK) {
    throw new Error(`Observation length ${obsBuffer.length}; expected ${OBS_SIZE * FRAME_STACK}.`);
  }
  if (hidden.length !== model.hiddenSize) {
    throw new Error(`Hidden-state length ${hidden.length}; expected ${model.hiddenSize}.`);
  }

  const tensor = new ort.Tensor("float32", obsBuffer, [1, obsBuffer.length]);
  const hiddenTensor = new ort.Tensor("float32", hidden, [1, 1, model.hiddenSize]);
  const results = await model.session.run({ observation: tensor, hidden_in: hiddenTensor });
  const m = Array.from(results.movement_logits.data as Float32Array);
  const c = Array.from(results.combat_logits.data as Float32Array);
  const t = Array.from(results.target_logits.data as Float32Array);
  const hiddenOut = new Float32Array(results.hidden_out.data as Float32Array);
  if (m.length !== MOVEMENT_ACTIONS || c.length !== COMBAT_ACTIONS || t.length !== TARGET_ACTIONS) {
    throw new Error(`Action head widths ${m.length}/${c.length}/${t.length}; expected 9/9/5.`);
  }
  if (hiddenOut.length !== model.hiddenSize) {
    throw new Error(`hidden_out length ${hiddenOut.length}; expected ${model.hiddenSize}.`);
  }
  return { m, c, t, hidden: hiddenOut };
}

// ═══════════════════════════════════════════════════════════════════
//  Frame Stack
// ═══════════════════════════════════════════════════════════════════
interface FrameStack {
  buf: Float32Array[];
  frameStack: number;
  idx: number;
  filled: boolean;
}

function createFrameStack(frameStack: number): FrameStack {
  const buf = new Array(frameStack).fill(null).map(() => new Float32Array(OBS_SIZE));
  return { buf, frameStack, idx: 0, filled: false };
}

function pushFrame(fs: FrameStack, obs: Float32Array) {
  if (fs.idx === 0) {
    for (let i = 0; i < fs.frameStack; i++) fs.buf[i] = new Float32Array(obs);
    fs.idx = fs.frameStack;
    fs.filled = true;
    return;
  }
  fs.buf[fs.idx % fs.frameStack] = new Float32Array(obs);
  fs.idx++;
  fs.filled = true;
}

function getStacked(fs: FrameStack): Float32Array {
  const out = new Float32Array(OBS_SIZE * fs.frameStack);
  for (let i = 0; i < fs.frameStack; i++) {
    const readIdx = fs.filled ? ((fs.idx - fs.frameStack + i) % fs.frameStack + fs.frameStack) % fs.frameStack : Math.min(i, fs.idx - 1);
    const srcIdx = Math.max(0, readIdx);
    out.set(fs.buf[srcIdx], i * OBS_SIZE);
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════
//  Canvas Renderer
// ═══════════════════════════════════════════════════════════════════
function render(ctx: CanvasRenderingContext2D, w: number, h: number, sim: SimState, playerPos: [number, number], overlays: any) {
  const scale = w / sim.arena;
  const cx = w / 2, cy = h / 2;
  const ts = (pos: [number, number]): [number, number] => [cx + pos[0] * scale, cy + pos[1] * scale];

  ctx.fillStyle = "#0a0e17"; ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = "rgba(40,60,90,0.25)"; ctx.lineWidth = 0.5;
  const gs = 500 * scale;
  for (let x = cx % gs; x < w; x += gs) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = cy % gs; y < h; y += gs) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

  // Area limits
  const half = sim.half * scale;
  ctx.strokeStyle = "rgba(80,120,180,0.35)"; ctx.lineWidth = 2;
  ctx.strokeRect(cx - half, cy - half, half * 2, half * 2);

  // Obstacles
  sim.obstacles.forEach(o => {
    const [sx, sy] = ts([o.x - o.hw, o.y - o.hh]);
    const sw = o.hw * 2 * scale, sh = o.hh * 2 * scale;
    const low = o.height < 300;
    ctx.fillStyle = low ? "rgba(120,100,60,0.6)" : "rgba(60,70,90,0.75)";
    ctx.fillRect(sx, sy, sw, sh);
    ctx.strokeStyle = low ? "rgba(180,150,80,0.4)" : "rgba(90,110,150,0.4)";
    ctx.lineWidth = 1; ctx.strokeRect(sx, sy, sw, sh);
  });

  // LOS indicator
  if (overlays.los) {
    const alive = sim.targets.filter(t => t.alive);
    if (alive.length > 0) {
      const t = alive[0];
      const [ax, ay] = ts(sim.agent.pos);
      const [bx, by] = ts(t.pos);
      const hasLOS = checkLOS(sim.agent.pos, t.pos, sim.obstacles);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      ctx.strokeStyle = hasLOS ? "rgba(100,255,100,0.2)" : "rgba(255,60,60,0.2)";
      ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  // Projectiles
  sim.projectiles.forEach(p => {
    const [sx, sy] = ts(p.pos);
    ctx.beginPath(); ctx.arc(sx, sy, p.isAgent ? 4 : 3, 0, Math.PI * 2);
    ctx.fillStyle = p.isAgent ? "#4af" : "#f64"; ctx.fill();

    const tLen = 8;
    const spd = Math.sqrt(p.vel[0] ** 2 + p.vel[1] ** 2) || 1;
    ctx.beginPath(); ctx.moveTo(sx, sy);
    ctx.lineTo(sx - p.vel[0] / spd * tLen * scale * 50, sy - p.vel[1] / spd * tLen * scale * 50);
    ctx.strokeStyle = p.isAgent ? "rgba(68,170,255,0.3)" : "rgba(255,100,68,0.3)";
    ctx.lineWidth = 2; ctx.stroke();
  });

  // Red Hostile Robots
  sim.targets.forEach(t => {
    if (!t.alive || t.isPlayerControlled) return;
    const [sx, sy] = ts(t.pos);
    const hf = t.hp / t.maxHp;
    ctx.beginPath(); ctx.arc(sx, sy, 12, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(220,60,60,${0.4 + hf * 0.6})`; ctx.fill();
    ctx.strokeStyle = "#e44"; ctx.lineWidth = 2; ctx.stroke();

    ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(sx - 16, sy - 22, 32, 5);
    ctx.fillStyle = hf > 0.3 ? "#e44" : "#f80"; ctx.fillRect(sx - 16, sy - 22, 32 * hf, 5);
    ctx.fillStyle = "#fff"; ctx.font = "10px monospace"; ctx.fillText(`T${t.id}`, sx - 6, sy + 24);
  });

  // Player
  const pl = sim.player;
  const [px, py] = ts(pl.pos);
  const plHf = pl.hp / pl.maxHp;
  const pw = pl.weapon;

  if (overlays.range) {
    ctx.beginPath(); ctx.arc(px, py, pw.range * scale, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(60,200,120,0.1)"; ctx.lineWidth = 1; ctx.stroke();
  }

  ctx.beginPath(); ctx.arc(px, py, 13, 0, Math.PI * 2);
  ctx.fillStyle = pw.isReloading ? "rgba(200,180,60,0.8)" : "rgba(60,200,120,0.8)";
  ctx.fill(); ctx.strokeStyle = "#4c8"; ctx.lineWidth = 2; ctx.stroke();

  ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(px - 16, py - 22, 32, 5);
  ctx.fillStyle = plHf > 0.3 ? "#4c8" : "#f80"; ctx.fillRect(px - 16, py - 22, 32 * plHf, 5);

  ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(px - 16, py - 15, 32, 3);
  ctx.fillStyle = pw.isReloading ? "#fc0" : "#8cf"; ctx.fillRect(px - 16, py - 15, 32 * (pw.ammo / pw.maxAmmo), 3);

  ctx.fillStyle = "#4c8"; ctx.font = "bold 10px monospace"; ctx.fillText("YOU", px - 10, py + 26);

  // Cyan AI Agent
  const ag = sim.agent;
  const [ax, ay] = ts(ag.pos);
  const hf = ag.hp / ag.maxHp;

  if (overlays.range) {
    const slot = ag.weapons[ag.activeWeapon];
    if (slot) {
      ctx.beginPath(); ctx.arc(ax, ay, slot.range * scale, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(68,170,255,0.12)"; ctx.lineWidth = 1; ctx.stroke();
    }
  }

  ctx.beginPath(); ctx.arc(ax, ay, 14, 0, Math.PI * 2);
  ctx.fillStyle = ag.isDodging ? "rgba(100,220,255,0.85)" :
    ag.lockRemain > 0 ? "rgba(255,200,60,0.7)" : `rgba(60,140,255,${0.5 + hf * 0.5})`;
  ctx.fill(); ctx.strokeStyle = "#4af"; ctx.lineWidth = 2; ctx.stroke();

  ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(ax - 18, ay - 26, 36, 5);
  ctx.fillStyle = hf > 0.3 ? "#4af" : "#f80"; ctx.fillRect(ax - 18, ay - 26, 36 * hf, 5);

  ctx.fillStyle = "#4af"; ctx.font = "bold 10px monospace"; ctx.fillText("AI", ax - 6, ay + 28);
}

// ═══════════════════════════════════════════════════════════════════
//  Main Component
// ═══════════════════════════════════════════════════════════════════
export default function CombatSandbox() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // High-performance mutable non-reactive state
  const simRef = useRef<SimState | null>(null);
  const playerPosRef = useRef<[number, number]>([500, 300]);
  const keysRef = useRef<Record<string, boolean>>({});
  const lastActionRef = useRef<[number, number, number]>([0, 0, 4]);
  const aiTickTimerRef = useRef<number>(0);
  const inferencePendingRef = useRef(false);
  const frameStackBufRef = useRef<FrameStack | null>(null);
  const playerActionsRef = useRef<{ fireTarget: [number, number] | null; reload: boolean }>({ fireTarget: null, reload: false });

  // Reactive state for HUD/Settings
  const [visualSim, setVisualSim] = useState<SimState | null>(null);
  const [lastActionState, setLastActionState] = useState<[number, number, number]>([0, 0, 4]);
  const [isPaused, setIsPaused] = useState(false);
  const [status, setStatus] = useState("No model loaded — AI uses scripted behavior");
  const [preset, setPreset] = useState<string>("scout");
  const [numTargets, setNumTargets] = useState(1);
  const [numObs, setNumObs] = useState(3);
  const [arenaSize, setArenaSize] = useState(2500);
  const [overlays, setOverlays] = useState({ facing: true, range: true, los: true });

  // C++ Alignment States for Registry/Decision / Sampling Modes
  const [selectedTier, setSelectedTier] = useState<"micro" | "small" | "medium" | "large">("medium");
  const [sampleStrategy, setSampleStrategy] = useState<"greedy" | "stochastic">("greedy");
  const [temperature, setTemperature] = useState<number>(1.0);

  // C++ Style Data Recording State (Part 2 Align)
  const [isRecording, setIsRecording] = useState(false);
  const [recordedStepCount, setRecordedStepCount] = useState(0);
  const recordedStepsRef = useRef<any[]>([]);
  const isRecordingRef = useRef(false);
  const encounterIdRef = useRef(1);

  const engagementStateRef = useRef<RewardEngagementState>({ stepsSinceDamage: 0, stepsSinceKill: 999, lowHpSeen: new Set<number>() });

  // New feature state: action probabilities, current observation, batch results
  const [currentLogits, setCurrentLogits] = useState<{ m: number[]; c: number[]; t: number[] } | null>(null);
  const [currentMasks, setCurrentMasks] = useState<{ m: boolean[]; c: boolean[]; t: boolean[] } | null>(null);
  const [currentObs, setCurrentObs] = useState<Float32Array | null>(null);
  const currentObsRef = useRef<Float32Array | null>(null);
  const [sidebarTab, setSidebarTab] = useState<"combat" | "tools">("combat");

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // Reward Logging state and refs
  const [rewardHistory, setRewardHistory] = useState<RewardStepLog[]>([]);
  const rewardHistoryRef = useRef<RewardStepLog[]>([]);
  const [stage, setStage] = useState<number>(3);
  const stageRef = useRef<number>(3);

  const accumRewardRef = useRef<number>(0);
  const accumBreakdownRef = useRef<any>({
    damageDealt: 0, killBonus: 0, damageTaken: 0, timePenalty: 0,
    optimalRange: 0, rangeClosing: 0, outOfRange: 0,
    flanking: 0, inactivity: 0, weaponSelection: 0, ammo: 0, endBonus: 0,
    engagement: 0,
  });

  useEffect(() => {
    stageRef.current = stage;
  }, [stage]);

  // Stable references for the animation frame loop
  const pausedRef = useRef(false);
  const overlaysRef = useRef(overlays);
  const modelInfoRef = useRef<LoadedModel | null>(null);

  const selectedTierRef = useRef<"micro" | "small" | "medium" | "large">("medium");
  const sampleStrategyRef = useRef<"greedy" | "stochastic">("greedy");
  const tempRef = useRef<number>(1.0);
  const prevTargetVelRef = useRef<Record<number, [number, number]>>({});

  // Sync state changes to refs immediately
  useEffect(() => { pausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { overlaysRef.current = overlays; }, [overlays]);
  useEffect(() => { selectedTierRef.current = selectedTier; }, [selectedTier]);
  useEffect(() => { sampleStrategyRef.current = sampleStrategy; }, [sampleStrategy]);
  useEffect(() => { tempRef.current = temperature; }, [temperature]);

  // Reset function
  const resetSim = useCallback(() => {
    encounterIdRef.current += 1;
    const activeStage = stageRef.current;
    const cfg = CURRICULUM_CONFIGS[activeStage] || CURRICULUM_CONFIGS[3];
    const s = createCurriculumSim(activeStage, true, encounterIdRef.current);
    simRef.current = s;
    playerPosRef.current = [s.player.pos[0], s.player.pos[1]];
    prevTargetVelRef.current = {};
    inferencePendingRef.current = false;
    lastActionRef.current = [0, 0, 4];
    setLastActionState([0, 0, 4]);
    setCurrentLogits(null);
    setCurrentMasks(null);
    setCurrentObs(null);
    if (modelInfoRef.current) {
      frameStackBufRef.current = createFrameStack(modelInfoRef.current.frameStack);
      modelInfoRef.current.hidden.fill(0);
    }

    rewardHistoryRef.current = [];
    setRewardHistory([]);
    engagementStateRef.current = { stepsSinceDamage: 0, stepsSinceKill: 999, lowHpSeen: new Set<number>() };
    accumRewardRef.current = 0;
    aiTickTimerRef.current = 0;

    // Mirror the authoritative stage configuration in the disabled debug fields.
    setPreset(cfg.preset);
    setNumTargets(cfg.targets);
    setNumObs(cfg.obstacles);
    setArenaSize(cfg.arena);
    setVisualSim(cloneSimState(s));
  }, []);

  const downloadRecordedCSV = () => {
    if (recordedStepsRef.current.length === 0) return;

    // 258 fields total: 5 metadata + 249 observation values + 3 actions + 1 reward.
    const csvHeaders = [
      "EncounterID", "EnemyName", "Archetype", "Frame", "CombatTime",
      ...Array.from({ length: 21 }, (_, i) => `Self_${i}`),
      ...Array.from({ length: 22 }, (_, i) => `Weapon_${i}`),
      ...Array.from({ length: 7 }, (_, i) => `Archetype_${i}`),
      ...Array.from({ length: 24 }, (_, i) => `PrimaryTarget_${i}`),
      ...Array.from({ length: 68 }, (_, i) => `Hostile_${i}`),
      ...Array.from({ length: 45 }, (_, i) => `Allied_${i}`),
      ...Array.from({ length: 8 }, (_, i) => `SpatialRing_${i}`),
      ...Array.from({ length: 8 }, (_, i) => `Cover_${i}`),
      ...Array.from({ length: 8 }, (_, i) => `ThreatSensing_${i}`),
      ...Array.from({ length: 9 }, (_, i) => `Navmesh_${i}`),
      ...Array.from({ length: 6 }, (_, i) => `GroupSummary_${i}`),
      "SpawnLeash",
      ...Array.from({ length: 7 }, (_, i) => `ExtendedThreat_${i}`),
      ...Array.from({ length: 4 }, (_, i) => `CanHit_${i}`),
      "TotalAmmo", "TargetsKilled",
      ...Array.from({ length: 4 }, (_, i) => `ArcClearance_${i}`),
      ...Array.from({ length: 5 }, (_, i) => `PlayerPattern_${i}`),
      "Action_Move", "Action_Combat", "Action_Target",
      "StepReward"
    ];

    const lines = [csvHeaders.join(",")];

    for (const step of recordedStepsRef.current) {
      if (!step.observation || step.observation.length < OBS_SIZE) continue;
      const row = [
        step.encounterId,
        `"${step.enemyName}"`,
        `"${step.archetype}"`,
        step.frame,
        step.combatTime.toFixed(3),
        ...step.observation.map((val: number) => val.toFixed(5)),
        step.action[0],
        step.action[1],
        step.action[2],
        step.reward.toFixed(4)
      ];
      lines.push(row.join(","));
    }

    const csvContent = lines.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `combat_sandbox_recording_encounter_${encounterIdRef.current}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Stage is authoritative: changing it rebuilds the same curriculum config
  // used by Python make_curriculum_env().
  useEffect(() => {
    stageRef.current = stage;
    resetSim();
  }, [stage, resetSim]);

  // Handle fire clicks on Canvas
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const currentSim = simRef.current;
    if (!currentSim || pausedRef.current) return;
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = 460 / rect.width, scaleY = 460 / rect.height;
    const screenX = (e.clientX - rect.left) * scaleX;
    const screenY = (e.clientY - rect.top) * scaleY;

    const worldX = (screenX - 230) / (460 / currentSim.arena);
    const worldY = (screenY - 230) / (460 / currentSim.arena);
    playerActionsRef.current.fireTarget = [worldX, worldY];
  }, []);

  // Keyboard listeners — Registered exactly ONCE on mount
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      keysRef.current[key] = true;
      if (key === ' ' || key === 'spacebar') {
        e.preventDefault();
        if (simRef.current?.agent) {
          playerActionsRef.current.fireTarget = [...simRef.current.agent.pos];
        }
      }
      if (key === 'r') {
        playerActionsRef.current.reload = true;
      }
      if (key === 'enter' && simRef.current?.done) {
        resetSim();
      }
    };
    const up = (e: KeyboardEvent) => {
      keysRef.current[e.key.toLowerCase()] = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // Set up canvas retina display bounds ONCE on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const size = 460;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
  }, []);

  // Render continuously, but advance the combat environment only on the
  // authoritative 0.2s decision cadence used by Python training.
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();

    const publishVisual = (sim: SimState, pp: [number, number]) => {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, 460, 460);
          render(ctx, 460, 460, sim, pp, overlaysRef.current);
        }
      }
      setVisualSim(cloneSimState(sim));
    };

    const commitDecision = (
      simAtDecision: SimState,
      obs: Float32Array,
      chosenAction: [number, number, number],
      pp: [number, number],
      pActions: { fireTarget: [number, number] | null; reload: boolean },
    ) => {
      // Ignore a stale async result if the episode was reset while ONNX ran.
      if (simRef.current !== simAtDecision) return;
      const prev = cloneSimState(simAtDecision);
      const next = tickSim(simAtDecision, chosenAction, pp, DT, pActions);
      const rewardData = calculateFrameReward(
        prev, next, chosenAction, stageRef.current, next.step, DT,
        engagementStateRef.current,
      );

      lastActionRef.current = chosenAction;
      setLastActionState(chosenAction);
      simRef.current = next;

      // Python extended env records target velocity before each transition;
      // the next observation uses current-prev to encode acceleration.
      prev.targets.forEach(t => {
        if (t.alive) prevTargetVelRef.current[t.id] = [t.vel[0], t.vel[1]];
      });

      const actionDesc = `Move[${chosenAction[0]}] + ${COMBAT_NAMES[chosenAction[1]] || "None"} (${chosenAction[2] === 4 ? "Keep" : `T${chosenAction[2]}`})`;
      const cumBefore = rewardHistoryRef.current[rewardHistoryRef.current.length - 1]?.cumReward || 0;
      const newLog: RewardStepLog = {
        step: next.step,
        action: chosenAction,
        actionDesc,
        reward: rewardData.reward,
        breakdown: rewardData.breakdown,
        cumReward: cumBefore + rewardData.reward,
      };
      rewardHistoryRef.current.push(newLog);
      if (rewardHistoryRef.current.length > 300) rewardHistoryRef.current.shift();
      setRewardHistory([...rewardHistoryRef.current]);

      if (isRecordingRef.current) {
        recordedStepsRef.current.push({
          encounterId: encounterIdRef.current,
          enemyName: TIER_PROFILES[selectedTierRef.current].name,
          archetype: selectedTierRef.current.toUpperCase(),
          frame: next.step,
          combatTime: next.agent.combatTime,
          observation: Array.from(obs),
          action: chosenAction,
          reward: rewardData.reward,
        });
        setRecordedStepCount(recordedStepsRef.current.length);
      }
      publishVisual(next, pp);
    };

    const loop = (time: number) => {
      animId = requestAnimationFrame(loop);
      let frameDt = (time - lastTime) / 1000;
      lastTime = time;
      if (frameDt <= 0) frameDt = 1 / 60;
      frameDt = Math.min(frameDt, 0.05);

      const currentSim = simRef.current;
      if (!currentSim) return;
      const pp = playerPosRef.current;

      if (!pausedRef.current && !currentSim.done) {
        // Human-controlled party member is sampled continuously, then its
        // position is consumed by the next 0.2s environment transition.
        const keys = keysRef.current;
        const playerSpeed = 350;
        let dx = 0, dy = 0;
        if (keys.w || keys.arrowup) dy -= 1;
        if (keys.s || keys.arrowdown) dy += 1;
        if (keys.a || keys.arrowleft) dx -= 1;
        if (keys.d || keys.arrowright) dx += 1;
        if (dx || dy) {
          const len = Math.hypot(dx, dy);
          pp[0] += dx / len * playerSpeed * frameDt;
          pp[1] += dy / len * playerSpeed * frameDt;
        }
        pp[0] = clamp(pp[0], -currentSim.half, currentSim.half);
        pp[1] = clamp(pp[1], -currentSim.half, currentSim.half);
        pushOutAABB(pp, currentSim.obstacles, 30);
        playerPosRef.current = [pp[0], pp[1]];

        aiTickTimerRef.current += frameDt;
        if (aiTickTimerRef.current >= DT && !inferencePendingRef.current) {
          aiTickTimerRef.current = Math.max(0, aiTickTimerRef.current - DT);
          const decisionSim = simRef.current;
          if (decisionSim && !decisionSim.done) {
            const pActions = { ...playerActionsRef.current };
            playerActionsRef.current = { fireTarget: null, reload: false };
            const decisionPlayerPos: [number, number] = [playerPosRef.current[0], playerPosRef.current[1]];
            const obs = buildObservation(decisionSim, decisionPlayerPos, prevTargetVelRef.current, DT);
            currentObsRef.current = obs;
            setCurrentObs(new Float32Array(obs));

            const mask = buildActionMask(decisionSim);
            setCurrentMasks(mask);

            // Frame stack advances on every environment transition, even on
            // action-lock ticks where ONNX/GRU inference is intentionally skipped.
            if (frameStackBufRef.current) pushFrame(frameStackBufRef.current, obs);

            if (mask.skipInference) {
              const lockedAction: [number, number, number] = [
                mask.m.findIndex(Boolean), mask.c.findIndex(Boolean), mask.t.findIndex(Boolean),
              ];
              commitDecision(decisionSim, obs, lockedAction, decisionPlayerPos, pActions);
            } else if (modelInfoRef.current?.session && frameStackBufRef.current) {
              inferencePendingRef.current = true;
              const activeModel = modelInfoRef.current;
              const stacked = getStacked(frameStackBufRef.current);
              runInference(activeModel, stacked, activeModel.hidden).then(logits => {
                if (modelInfoRef.current !== activeModel || simRef.current !== decisionSim) return;
                activeModel.hidden = logits.hidden;
                setCurrentLogits(logits);
                const currentMask = buildActionMask(decisionSim);
                setCurrentMasks(currentMask);
                const chosen: [number, number, number] = sampleStrategyRef.current === "stochastic"
                  ? [
                      softmaxSample(logits.m, currentMask.m, tempRef.current),
                      softmaxSample(logits.c, currentMask.c, tempRef.current),
                      softmaxSample(logits.t, currentMask.t, tempRef.current),
                    ]
                  : [
                      argmaxMasked(logits.m, currentMask.m),
                      argmaxMasked(logits.c, currentMask.c),
                      argmaxMasked(logits.t, currentMask.t),
                    ];
                commitDecision(decisionSim, obs, chosen, decisionPlayerPos, pActions);
              }).catch(err => console.error("ONNX model run error:", err))
                .finally(() => { inferencePendingRef.current = false; });
            } else {
              const fallback = scriptedAI(decisionSim, decisionPlayerPos);
              commitDecision(decisionSim, obs, fallback, decisionPlayerPos, pActions);
            }
          }
        }
      }

      // Rendering is decoupled from simulation stepping.
      const renderSim = simRef.current;
      if (renderSim) {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, 460, 460);
            render(ctx, 460, 460, renderSim, playerPosRef.current, overlaysRef.current);
          }
        }
      }
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Load only the current named recurrent ONNX contract. Behavioural tier is
  // selected explicitly because all four tiers share the same tensor shapes.
  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    setStatus("Loading ONNX Model...");
    try {
      const info = await loadOnnxModel(file);
      modelInfoRef.current = info;
      frameStackBufRef.current = createFrameStack(info.frameStack);

      setStatus(`Loaded contract-compatible ONNX model (249 × ${info.frameStack}, hidden ${info.hiddenSize}). Tier remains ${selectedTier.toUpperCase()} because tensor shapes do not encode behavioural tier.`);
    } catch (err: any) {
      setStatus(`Error: ${err.message}`);
      console.error(err);
    }
  };

  const ag = visualSim?.agent;
  const lastAction = lastActionState;
  const activeFrameStack = modelInfoRef.current ? modelInfoRef.current.frameStack : TIER_PROFILES[selectedTier].frameStack;

  // Batch episode runner callback
  const handleRunBatch = useCallback(async (n: number): Promise<BatchResults> => {
    const model = modelInfoRef.current;
    if (!model) return { episodes: [], winRate: 0, meanReward: 0, stdReward: 0, meanKills: 0, meanLength: 0, rewardByOutcome: { wins: 0, losses: 0 } };

    const episodes: EpisodeResult[] = [];
    for (let ep = 0; ep < n; ep++) {
      const sim = createCurriculumSim(stage, false, 42 + ep);
      const fs = createFrameStack(model.frameStack);
      const prevVelMap: Record<number, [number, number]> = {};
      const engState: RewardEngagementState = { stepsSinceDamage: 0, stepsSinceKill: 999, lowHpSeen: new Set<number>() };
      let cumReward = 0;
      let kills = 0;
      let hidden = new Float32Array(model.hiddenSize);
      const fixedPlayerPos: [number, number] = [sim.arena * 0.2, sim.arena * 0.12];

      while (!sim.done) {
        const obs = buildObservation(sim, fixedPlayerPos, prevVelMap, DT);
        pushFrame(fs, obs);
        const mask = buildActionMask(sim);
        let action: [number, number, number];

        if (mask.skipInference) {
          // Recurrent state is intentionally frozen on action-lock ticks.
          action = [mask.m.findIndex(Boolean), mask.c.findIndex(Boolean), mask.t.findIndex(Boolean)];
        } else {
          const logits = await runInference(model, getStacked(fs), hidden);
          hidden = logits.hidden;
          action = [
            argmaxMasked(logits.m, mask.m),
            argmaxMasked(logits.c, mask.c),
            argmaxMasked(logits.t, mask.t),
          ];
        }

        const prev = cloneSimState(sim);
        // Save pre-transition velocities exactly when Python ExtendedEnv.step()
        // does; the next observation uses them for acceleration.
        for (const t of sim.targets) if (t.alive) prevVelMap[t.id] = [t.vel[0], t.vel[1]];
        const prevAlive = sim.targets.filter(t => t.alive).length;
        tickSim(sim, action, fixedPlayerPos, DT, {});
        const currAlive = sim.targets.filter(t => t.alive).length;
        kills += Math.max(0, prevAlive - currAlive);
        cumReward += calculateFrameReward(prev, sim, action, stage, sim.step, DT, engState).reward;
      }

      const win = sim.agent.hp > 0 && sim.targets.every(t => !t.alive);
      episodes.push({ win, reward: cumReward, kills, length: sim.step });
    }

    return { episodes } as BatchResults;
  }, [stage]);

  return (
    <div style={{ background: "#0d1117", color: "#c9d1d9", minHeight: "100vh", fontFamily: "'JetBrains Mono','Fira Code',monospace", fontSize: 12 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", borderBottom: "1px solid #21262d", background: "#161b22" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: "#58a6ff" }}>⚔ Combat Sandbox</span>
          <span style={{ color: modelInfoRef.current ? "#7ee787" : "#f0883e" }}>{status}</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label style={{ padding: "4px 12px", background: "#238636", color: "#fff", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 12 }}>
            Load ONNX Model
            <input type="file" accept=".onnx" onChange={handleModelUpload} style={{ display: "none" }} />
          </label>
          <button onClick={() => resetSim()} style={btn}>Reset</button>
          <button onClick={() => setIsPaused(!isPaused)} style={{ ...btn, background: isPaused ? "#238636" : "#da3633" }}>
            {isPaused ? "▶ Resume" : "⏸ Pause"}
          </button>
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 48px)", overflow: "hidden" }}>

          {/* ── LEFT: Canvas + Controls ── */}
          <div style={{ width: 476, padding: "8px 8px 8px 12px", flexShrink: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ position: "relative", width: 460, height: 460 }}>
              <canvas ref={canvasRef} onClick={handleCanvasClick}
                style={{ width: 460, height: 460, borderRadius: 6, border: "1px solid #21262d", cursor: "crosshair" }} />

              {/* Game Over Overlay */}
              {visualSim?.done && (() => {
                const hostiles = visualSim.targets;
                const allHostilesDead = hostiles.length > 0 && hostiles.every(t => !t.alive);
                const agentDead = visualSim.agent.hp <= 0;
                const playerDead = visualSim.player.hp <= 0;
                let outcome: string, outcomeColor: string, subtext: string;
                if (allHostilesDead) { outcome = "AI WINS"; outcomeColor = "#7ee787"; subtext = "All hostile targets eliminated"; }
                else if (playerDead) { outcome = "YOU DIED"; outcomeColor = "#f85149"; subtext = "The AI agent eliminated you"; }
                else if (agentDead) { outcome = "YOU WIN"; outcomeColor = "#58a6ff"; subtext = "AI agent destroyed"; }
                else { outcome = "TIMEOUT"; outcomeColor = "#ffa657"; subtext = `Step limit reached (${visualSim.maxSteps})`; }
                const kills = hostiles.filter(t => !t.alive).length;
                const cumRew = rewardHistory.length > 0 ? rewardHistory[rewardHistory.length - 1].cumReward : 0;
                return (
                  <div style={{ position: "absolute", inset: 0, borderRadius: 6, background: "rgba(0,0,0,0.78)", backdropFilter: "blur(4px)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", zIndex: 20 }}>
                    <div style={{ fontSize: 32, fontWeight: 900, color: outcomeColor, textShadow: `0 0 24px ${outcomeColor}40`, letterSpacing: 3, marginBottom: 4 }}>{outcome}</div>
                    <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 16 }}>{subtext}</div>
                    <div style={{ display: "flex", gap: 20, marginBottom: 18, padding: "10px 16px", background: "rgba(255,255,255,0.04)", borderRadius: 6, border: "1px solid rgba(255,255,255,0.06)" }}>
                      {[{ v: `${kills}/${hostiles.length}`, l: "Kills", c: "#c9d1d9" }, { v: cumRew.toFixed(0), l: "Reward", c: "#58a6ff" }, { v: `${visualSim.step}`, l: "Steps", c: "#c9d1d9" }].map((s, i) => (
                        <div key={i} style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 20, fontWeight: 700, color: s.c, fontFamily: "monospace" }}>{s.v}</div>
                          <div style={{ fontSize: 8, color: "#8b949e" }}>{s.l}</div>
                        </div>
                      ))}
                    </div>
                    <button onClick={() => resetSim()} style={{ padding: "8px 28px", fontSize: 13, fontWeight: 700, fontFamily: "inherit", background: outcomeColor, color: "#0d1117", border: "none", borderRadius: 6, cursor: "pointer" }}>RESTART</button>
                    <div style={{ color: "#6e7681", fontSize: 9, marginTop: 6 }}>or press Enter</div>
                  </div>
                );
              })()}
            </div>

            {/* Controls row */}
            <div style={{ display: "flex", gap: 8, marginTop: 6, alignItems: "center", flexWrap: "wrap" }}>
              {(["facing", "range", "los"] as const).map(k => (
                <label key={k} style={{ display: "flex", alignItems: "center", gap: 3, color: "#6e7681", cursor: "pointer", fontSize: 10 }}>
                  <input type="checkbox" checked={overlays[k]} onChange={() => setOverlays(p => ({ ...p, [k]: !p[k] }))} style={{ width: 12, height: 12 }} /> {k}
                </label>
              ))}
              <span style={{ color: "#484848", fontSize: 9 }}>·</span>
              <span style={{ fontSize: 9, color: "#6e7681" }}>
                <span style={{ color: "#58a6ff" }}>WASD</span> move · <span style={{ color: "#ffa657" }}>Click</span>/<span style={{ color: "#ffa657" }}>Space</span> shoot · <span style={{ color: "#fc0" }}>R</span> reload
              </span>
            </div>

            {/* Quick status bar */}
            {visualSim && (
              <div style={{ display: "flex", gap: 10, marginTop: 6, padding: "5px 8px", background: "#161b22", borderRadius: 4, border: "1px solid #21262d", fontSize: 10 }}>
                <span style={{ color: "#8b949e" }}>Step <strong style={{ color: "#c9d1d9" }}>{visualSim.step}</strong>/{visualSim.maxSteps}</span>
                <span style={{ color: "#8b949e" }}>Act: <span style={{ color: "#7ee787" }}>M{lastAction[0]}</span> <span style={{ color: "#ffa657" }}>{COMBAT_NAMES[lastAction[1]]}</span> <span style={{ color: "#d2a8ff" }}>T{lastAction[2]}</span></span>
                {visualSim.done && <span style={{ color: visualSim.targets.every(t => !t.alive) ? "#7ee787" : "#f85149", fontWeight: 700 }}>
                  {visualSim.targets.every(t => !t.alive) ? "AI Wins" : visualSim.agent.hp <= 0 ? "You Win" : "Timeout"}
                </span>}
              </div>
            )}

            {/* D3 Reward Chart — under the canvas */}
            <RewardD3Chart history={rewardHistory} stage={stage} />
          </div>

          {/* ── RIGHT: Tabbed Panel ── */}
          <div style={{ flex: 1, borderLeft: "1px solid #21262d", display: "flex", flexDirection: "column", minWidth: 320 }}>
            {/* Tab bar */}
            <div style={{ display: "flex", borderBottom: "1px solid #21262d", background: "#161b22", flexShrink: 0 }}>
              {(["combat", "tools"] as const).map(tab => (
                <button key={tab} onClick={() => setSidebarTab(tab as any)} style={{
                  flex: 1, padding: "7px 0", fontSize: 11, fontWeight: sidebarTab === tab ? 700 : 400, fontFamily: "inherit",
                  background: sidebarTab === tab ? "#0d1117" : "transparent",
                  color: sidebarTab === tab ? "#58a6ff" : "#8b949e",
                  border: "none", borderBottom: sidebarTab === tab ? "2px solid #58a6ff" : "2px solid transparent",
                  cursor: "pointer", textTransform: "uppercase", letterSpacing: 1,
                }}>
                  {tab === "combat" ? "⚔ Combat" : "🔧 Tools"}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div style={{ flex: 1, overflow: "auto", padding: 10 }}>

              {/* ════ COMBAT TAB (includes config) ════ */}
              {sidebarTab === "combat" && (
                <div>
                  {/* Config section */}
                  <div style={{ marginBottom: 10, padding: "8px 10px", background: "#161b22", borderRadius: 6, border: "1px solid #21262d" }}>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px 12px", marginBottom: 8 }}>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Weapon
                        <select value={preset} disabled style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0, opacity: 0.75 }}><option value="heavy">Heavy</option><option value="scout">Scout</option><option value="sniper">Sniper</option><option value="tank">Tank</option><option value="melee_bot">Melee Bot</option></select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Arena Size
                        <select value={arenaSize} disabled style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[2000, 2500, 3000, 3500, 4000, 5000].map(s => <option key={s} value={s}>{s}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Stage
                        <select value={stage} onChange={e => setStage(+e.target.value)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[1, 2, 3, 4, 5, 6, 7].map(n => <option key={n} value={n}>{n}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Targets
                        <select value={numTargets} disabled style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[0, 1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Obstacles
                        <select value={numObs} disabled style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[0, 2, 4, 6, 8, 12, 16].map(n => <option key={n} value={n}>{n}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Tier Profile
                        <select value={selectedTier} onChange={e => setSelectedTier(e.target.value as any)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}><option value="micro">Micro (5Hz)</option><option value="small">Small (5Hz)</option><option value="medium">Medium (5Hz)</option><option value="large">Large (5Hz)</option></select>
                      </label>
                    </div>
                    <div style={{ display: "flex", gap: 12, alignItems: "center", borderTop: "1px solid #21262d", paddingTop: 6 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ color: "#8b949e", fontSize: 11 }}>Action Picker:</span>
                        <select value={sampleStrategy} onChange={e => setSampleStrategy(e.target.value as any)} style={sel}><option value="greedy">Argmax (Greedy)</option><option value="stochastic">Softmax (Stochastic)</option></select>
                      </div>
                      {sampleStrategy === "stochastic" && (
                        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <span style={{ color: "#8b949e", fontSize: 11 }}>Temp:</span>
                          <input type="range" min="0.1" max="2.0" step="0.1" value={temperature} onChange={e => setTemperature(+e.target.value)} style={{ width: 70, accentColor: "#58a6ff" }} />
                          <span style={{ color: "#c9d1d9", fontSize: 10 }}>{temperature.toFixed(1)}</span>
                        </div>
                      )}
                      <span style={{ fontSize: 9, color: "#6e7681", marginLeft: "auto" }}>
                        {TIER_PROFILES[selectedTier].decisionInterval}s · {activeFrameStack} frames · dim {OBS_SIZE * activeFrameStack}
                      </span>
                    </div>
                  </div>

                  {/* Two-column combat metrics */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, alignItems: "start" }}>
                  {/* Col 1: HP bars + targets */}
                  <div>
                    {visualSim?.player && <>
                      <div style={sec}>Player</div>
                      <Row label="HP"><Bar v={visualSim.player.hp / visualSim.player.maxHp} c="#4c8" /><span>{visualSim.player.hp.toFixed(0)}/{visualSim.player.maxHp}</span></Row>
                      <Row label={visualSim.player.weapon.name}>
                        <Bar v={visualSim.player.weapon.ammo / visualSim.player.weapon.maxAmmo} c={visualSim.player.weapon.isReloading ? "#fc0" : "#8cf"} />
                        <span>{visualSim.player.weapon.ammo}/{visualSim.player.weapon.maxAmmo}{visualSim.player.weapon.isReloading ? " ⟳" : ""}</span>
                      </Row>
                    </>}
                    {ag && <>
                      <div style={{ ...sec, marginTop: 8 }}>AI Agent</div>
                      <Row label="HP"><Bar v={ag.hp / ag.maxHp} c="#4af" /><span>{ag.hp.toFixed(0)}/{ag.maxHp}</span></Row>
                      {ag.weapons.map((w, i) => (
                        <Row key={i} label={`${w.name}${i === ag.activeWeapon ? " ◄" : ""}`}>
                          <Bar v={w.ammo / w.maxAmmo} c={w.isReloading ? "#fc0" : i === ag.activeWeapon ? "#7ee787" : "#555"} />
                          <span>{w.ammo}/{w.maxAmmo}{w.isReloading ? " ⟳" : ""}</span>
                        </Row>
                      ))}
                      {ag.lockRemain > 0 && <div style={{ color: "#fc0", fontSize: 10 }}>🔒 {LOCK_NAMES[ag.lockReason]} ({((1 - ag.lockRemain / ag.lockDuration) * 100).toFixed(0)}%)</div>}
                      {ag.isDodging && <div style={{ color: "#4df", fontSize: 10 }}>⚡ Dodging</div>}
                    </>}
                    {visualSim && <>
                      <div style={{ ...sec, marginTop: 8 }}>Targets</div>
                      {visualSim.targets.filter(t => !t.isPlayerControlled).map(t => (
                        <Row key={t.id} label={`T${t.id} ${t.role}`} style={{ opacity: t.alive ? 1 : 0.3 }}>
                          <Bar v={t.hp / t.maxHp} c="#e44" /><span>{t.hp.toFixed(0)}/{t.maxHp}{!t.alive ? " ☠" : ""}</span>
                        </Row>
                      ))}
                    </>}
                  </div>
                  {/* Col 2: Action probs + reward budget */}
                  <div>
                    <ActionProbabilityHeatmap logits={currentLogits} masks={currentMasks} chosenAction={lastAction as [number, number, number]} />
                    {visualSim && (
                      <div style={{ marginTop: 8 }}>
                        <RewardBudgetBar
                          cumReward={rewardHistory.length > 0 ? rewardHistory[rewardHistory.length - 1].cumReward : 0}
                          step={visualSim.step} maxSteps={visualSim.maxSteps}
                          numTargets={visualSim.targets.length}
                          kills={visualSim.targets.filter(t => !t.alive).length}
                          agentAlive={visualSim.agent.hp > 0} done={visualSim.done} stage={stage}
                        />
                      </div>
                    )}
                  </div>
                </div>
                </div>
              )}

              {/* ════ TOOLS TAB ════ */}
              {sidebarTab === "tools" && (
                <div>
                  <div style={sec}>Data Recorder</div>
                  <div style={{ background: "#161b22", padding: 10, borderRadius: 6, border: "1px solid #21262d", marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <span style={{ color: "#8b949e" }}>Status:</span>
                      <span style={{ color: isRecording ? "#7ee787" : "#8b949e", fontWeight: "bold" }}>{isRecording ? "● REC" : "○ Idle"}</span>
                    </div>
                    <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
                      <button onClick={() => setIsRecording(prev => !prev)} style={{ ...btn, flex: 1, background: isRecording ? "#341d1d" : "#1f2328", borderColor: isRecording ? "#9c2b2b" : "#44464c", color: isRecording ? "#ff8c8c" : "#c9d1d9" }}>{isRecording ? "⏸ Pause" : "⏺ Record"}</button>
                      <button onClick={() => { recordedStepsRef.current = []; setRecordedStepCount(0); }} style={btn} disabled={isRecording}>Clear</button>
                    </div>
                    <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 6 }}>📊 {recordedStepCount} steps · Enc #{encounterIdRef.current}</div>
                    <button onClick={downloadRecordedCSV} disabled={recordedStepCount === 0} style={{ ...btn, width: "100%", background: recordedStepCount > 0 ? "#238636" : "#21262d", color: "#fff", opacity: recordedStepCount > 0 ? 1 : 0.5 }}>📥 Export CSV</button>
                  </div>
                  <div style={{ marginBottom: 10 }}><BatchEpisodeRunner onRunBatch={handleRunBatch} disabled={!modelInfoRef.current} /></div>
                  <ObservationGroupInspector obs={currentObs} />
                </div>
              )}
            </div>
          </div>
      </div>
    </div>
  );
}

// Fallback scripted agent behavior
function scriptedAI(sim: SimState, playerPos: [number, number]): [number, number, number] {
  const ag = sim.agent;
  const slot = ag.weapons[ag.activeWeapon];
  const alive = sim.targets.filter(t => t.alive);
  if (!alive.length) return [0, 0, 4];
  const t = alive[0];
  const d = dist(ag.pos, t.pos);
  const toT = norm([t.pos[0] - ag.pos[0], t.pos[1] - ag.pos[1]]);

  let mIdx = 0;
  const optMid = slot ? ((slot.optMin || 0) + (slot.optMax || slot.range)) / 2 : 800;
  const approach = d > optMid + 100 ? 1 : d < optMid - 100 ? -1 : 0;
  const perpDir = [-toT[1], toT[0]];
  const moveDir: [number, number] = [toT[0] * approach * 0.6 + perpDir[0] * 0.4, toT[1] * approach * 0.6 + perpDir[1] * 0.4];
  let bestDot = -2;
  const desired = norm(moveDir);
  for (let i = 0; i < MOVEMENT_ACTIONS; i++) {
    const candidate = movementDirection(sim, i);
    const dd = dot(candidate, desired);
    if (dd > bestDot) { bestDot = dd; mIdx = i; }
  }

  let cIdx = 0;
  if (slot && slot.ammo > 0 && slot.cdRemain <= 0 && !slot.isReloading && d <= slot.range) cIdx = 1;
  else if (slot && slot.ammo <= 0 && !slot.isReloading) cIdx = 2;
  else if (slot && slot.isReloading && ag.weapons.length > 1) {
    const other = 1 - ag.activeWeapon;
    if (ag.weapons[other].ammo > 0) cIdx = other === 0 ? 3 : 4;
  }

  return [mIdx, cIdx, 0];
}

function argmaxMasked(logits: number[], mask: boolean[]): number {
  let best = -Infinity, idx = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = mask[i] ? logits[i] : -1e8;
    if (v > best) { best = v; idx = i; }
  }
  return idx;
}

// UI layout shared structures & styles
const btn = { padding: "4px 10px", background: "#21262d", color: "#c9d1d9", border: "1px solid #30363d", borderRadius: 6, cursor: "pointer", fontSize: 12, fontFamily: "inherit" };
const sel = { marginLeft: 4, padding: "2px 4px", background: "#0d1117", color: "#c9d1d9", border: "1px solid #30363d", borderRadius: 4, fontSize: 12, fontFamily: "inherit" };
const sec = { color: "#58a6ff", fontWeight: 700, marginBottom: 6, fontSize: 13, borderBottom: "1px solid #21262d", paddingBottom: 4 };

function Row({ label, children, style }: { label: string; children: React.ReactNode; style?: React.CSSProperties; key?: any }) {
  return <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3, ...style }}><span style={{ minWidth: 80 }}>{label}</span>{children}</div>;
}

function Bar({ v, c }: { v: number; c: string }) {
  return <div style={{ flex: 1, height: 8, background: "#21262d", borderRadius: 4, overflow: "hidden" }}>
    <div style={{ width: `${clamp(v * 100, 0, 100)}%`, height: "100%", background: c, borderRadius: 4, transition: "width 0.15s" }} />
  </div>;
}
