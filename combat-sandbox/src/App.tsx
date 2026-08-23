import React, { useState, useEffect, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";
import { RewardD3Chart, RewardStepLog } from "./components/RewardD3Chart";
import { ActionProbabilityHeatmap } from "./components/ActionProbabilityHeatmap";
import { ObservationGroupInspector } from "./components/ObservationGroupInspector";
import { RewardBudgetBar } from "./components/RewardBudgetBar";
import { BatchEpisodeRunner, EpisodeResult, BatchResults } from "./components/BatchEpisodeRunner";

// Configure WASM paths for onnxruntime-web
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
ort.env.wasm.proxy = true;

// ═══════════════════════════════════════════════════════════════════
//  Constants (match C++ / Python exactly)
// ═══════════════════════════════════════════════════════════════════
const OBS_SIZE = 249;
const MOVEMENT_ACTIONS = 9;
const COMBAT_ACTIONS = 9;
const TARGET_ACTIONS = 5;
const DT = 0.2; // decision interval
const PHYSICS_DT = 1 / 60;
const S = 0.7071067811865476; // 1/√2
const MOVE_DIRS = [[0, 0], [0, -1], [S, -S], [1, 0], [S, S], [0, 1], [-S, S], [-1, 0], [-S, -S]];
const SPATIAL_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315];
const LOCK_NAMES = ["", "Firing", "Reloading", "Dodging", "Melee", "Switching", "WindUp", "Repositioning"];
const COMBAT_NAMES = ["None", "Fire", "Reload", "Sw0", "Sw1", "Melee", "Block", "Dodge", "Reposition"];
const AGENT_BODY_RADIUS = 30;
const DEFENCE_CONSTANT = 100;
const MIN_DAMAGE = 1;

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
    canArc: boolean;
    maxArcHeight?: number;
  }[];
  melee: { damage: number; range: number; cooldown: number };
}

const WEAPON_PRESETS: Record<string, WeaponPreset> = {
  heavy: {
    slots: [
      { name: "Cannon", baseDmg: 35, range: 2000, maxAmmo: 6, fireCd: 1.0, reloadTime: 3.0, windUp: 0.5, projSpeed: 2000, optMin: 800, optMax: 1600, canArc: false },
      { name: "Missiles", baseDmg: 25, range: 1800, maxAmmo: 4, fireCd: 1.5, reloadTime: 4.0, projSpeed: 1200, optMin: 600, optMax: 1400, canArc: true, maxArcHeight: 400 },
    ],
    melee: { damage: 40, range: 250, cooldown: 1.5 },
  },
  scout: {
    slots: [
      { name: "Laser", baseDmg: 8, range: 1200, maxAmmo: 20, fireCd: 0.2, reloadTime: 2.0, projSpeed: 4500, optMin: 400, optMax: 900, canArc: false },
    ],
    melee: { damage: 15, range: 150, cooldown: 0.8 },
  },
  sniper: {
    slots: [
      { name: "Railgun", baseDmg: 80, range: 3000, maxAmmo: 1, fireCd: 2.0, reloadTime: 3.0, windUp: 1.0, projSpeed: 6000, optMin: 1500, optMax: 2800, canArc: false },
      { name: "Sidearm", baseDmg: 10, range: 1000, maxAmmo: 12, fireCd: 0.3, reloadTime: 2.0, projSpeed: 3500, optMin: 300, optMax: 800, canArc: false },
    ],
    melee: { damage: 12, range: 150, cooldown: 1.0 },
  },
  tank: {
    slots: [
      { name: "Gatling", baseDmg: 5, range: 1500, maxAmmo: 100, fireCd: 0.08, reloadTime: 4.0, projSpeed: 4000, optMin: 400, optMax: 1200, canArc: false },
    ],
    melee: { damage: 30, range: 200, cooldown: 1.2 },
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
  canArc: boolean;
  maxArcHeight?: number;
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
  role: string;
  defence: number;
  barrier: number;
  atkCd: number;
  projSpeed: number;
  atkDmg: number;
  atkRange: number;
  atkStat: number;
  critChance: number;
  critMult: number;
  characterType?: number;
  mana?: number;
  maxMana?: number;
  commitment?: number;
  gapCloserCd?: number;
  gapCloserCooldown?: number;
  gapCloserRange?: number;
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
  impactRadius?: number;
  arcStart?: [number, number];
  arcApex?: [number, number];
  arcEnd?: [number, number];
  arcHeight?: number;
  arcFlightTime?: number;
  arcElapsed?: number;
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
  dodgeDistance: number;
  dodgeEaseExponent: number;
  dodgeInvulnDuration: number;
  dodgeElapsed: number;
  dodgeStartPos: [number, number];
  dodgeEndPos: [number, number];
  isBlocking: boolean;
  blockDefenceBonus: number;
  blockMoveMultiplier: number;
  isRepositioning: boolean;
  repositionRemain: number;
  repositionDuration: number;
  repositionCd: number;
  repositionCooldown: number;
  repositionSpeedMultiplier: number;
  isSwitching: boolean;
  switchRemain: number;
  switchTarget: number;
  switchTime: number;
  isWindingUp: boolean;
  windUpRemain: number;
  pendingFire: { targetPos: [number, number]; slotIdx: number } | null;
  lockRemain: number;
  lockDuration: number;
  lockReason: number;
  combatTime: number;
  maxSpeed: number;
  spawnPos: [number, number];
  leashRange: number;
  activeTargetIdx: number;
  cachedMovementAction: number;
  archetype: "ranged" | "melee" | "healer" | "tank";
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
  obstacles: Obstacle[];
  projectiles: Projectile[];
  player: PlayerState;
  step: number;
  maxSteps: number;
  done: boolean;
  patterns: {
    aggression: number;
    preferredRange: number;
    evasion: number;
    manaBurn: number;
    directionEntropy: number;
    directionCounts: number[];
  };
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

function segPointDist(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  if (len2 < 0.01) return Math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2);
  const t = clamp(((cx - ax) * dx + (cy - ay) * dy) / len2, 0, 1);
  return Math.sqrt((cx - (ax + t * dx)) ** 2 + (cy - (ay + t * dy)) ** 2);
}

function computeDamage(baseDmg: number, atkStat: number, defence: number, barrier = 0, critChance = 0, critMult = 1.5) {
  let outgoing = baseDmg + atkStat;
  const wasCrit = Math.random() < critChance;
  if (wasCrit) outgoing *= critMult;
  const barrierAbsorbed = Math.min(barrier, outgoing);
  const remaining = outgoing - barrierAbsorbed;
  const newBarrier = barrier - barrierAbsorbed;
  const mitigation = defence / (defence + DEFENCE_CONSTANT);
  const hpDamage = remaining > 0 ? Math.max(MIN_DAMAGE, remaining * (1 - mitigation)) : 0;
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

function blockingObstacleHeight(a: [number, number], b: [number, number], obstacles: Obstacle[]): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  let nearestT = Infinity, height = 0;
  for (const o of obstacles) {
    const t = rayAABB(a[0], a[1], dx, dy, o.x, o.y, o.hw, o.hh);
    if (t !== null && t < nearestT) { nearestT = t; height = o.height; }
  }
  return height;
}

function weaponHasPath(slot: Pick<Weapon, "canArc" | "maxArcHeight">, a: [number, number], b: [number, number], obstacles: Obstacle[]): boolean {
  if (checkLOS(a, b, obstacles)) return true;
  const height = blockingObstacleHeight(a, b, obstacles);
  return slot.canArc && height > 0 && ((slot.maxArcHeight || 0) <= 0 || height <= (slot.maxArcHeight || 0));
}

function pushOutAABB(pos: [number, number], obstacles: Obstacle[], radius = 20): [number, number] {
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

// ═══════════════════════════════════════════════════════════════════
//  Sim State Factory
// ═══════════════════════════════════════════════════════════════════
function createSim(presetName: string = "heavy", arenaSize = 2500, numTargets = 2, numObstacles = 4, archetype?: AgentState["archetype"]): SimState {
  const half = arenaSize * 0.45;
  const wp = WEAPON_PRESETS[presetName] || WEAPON_PRESETS.heavy;
  const rp = () => (Math.random() - 0.5) * half * 1.5;

  const weapons: Weapon[] = wp.slots.map(s => ({
    ...s, ammo: s.maxAmmo, cdRemain: 0, reloadRemain: 0, isReloading: false,
  }));

  const obstacles: Obstacle[] = [];
  for (let i = 0; i < numObstacles; i++) {
    const hw = 40 + Math.random() * 60, hh = 40 + Math.random() * 60;
    obstacles.push({ x: rp(), y: rp(), hw, hh, height: Math.random() > 0.4 ? 300 : 150 });
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
    role: "player",
    defence: 20,
    barrier: 0,
    atkCd: 0,
    projSpeed: 3500,
    atkDmg: 15,
    atkRange: 1500,
    atkStat: 8,
    critChance: 0.05,
    critMult: 1.5,
    characterType: 0.4, mana: 100, maxMana: 100, commitment: 0,
    gapCloserCd: 0, gapCloserCooldown: 5, gapCloserRange: 500,
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
      defence: 30,
      barrier: 0,
      atkCd: 0,
      projSpeed: 2500,
      atkDmg: 8,
      atkRange: 1200,
      atkStat: 5,
      critChance: 0.05,
      critMult: 1.5,
      characterType: 0.4, mana: 100, maxMana: 100, commitment: 0,
      gapCloserCd: 0, gapCloserCooldown: 5, gapCloserRange: 500,
    });
  }

  return {
    arena: arenaSize, half, preset: wp,
    agent: {
      pos: [rp() * 0.3, rp() * 0.3], vel: [0, 0], facing: [1, 0], hp: 100, maxHp: 100,
      barrier: 0, defence: 25, atkStat: 10, critChance: 0.05, critMult: 1.5,
      weapons, activeWeapon: 0, melee: { ...wp.melee, cdRemain: 0 },
      isDodging: false, dodgeRemain: 0, dodgeCd: 0, dodgeDir: [0, 0], dodgeDuration: 0.35, dodgeCooldown: 2.5,
      dodgeDistance: 400, dodgeEaseExponent: 2, dodgeInvulnDuration: 0.15, dodgeElapsed: 0,
      dodgeStartPos: [0, 0], dodgeEndPos: [0, 0],
      isBlocking: false, blockDefenceBonus: 80, blockMoveMultiplier: 0.3,
      isRepositioning: false, repositionRemain: 0, repositionDuration: 0.6,
      repositionCd: 0, repositionCooldown: 3, repositionSpeedMultiplier: 1.75,
      isSwitching: false, switchRemain: 0, switchTarget: 0, switchTime: 0.2,
      isWindingUp: false, windUpRemain: 0, pendingFire: null,
      lockRemain: 0, lockDuration: 0, lockReason: 0, combatTime: 0,
      maxSpeed: 450, spawnPos: [0, 0], leashRange: 2000,
      activeTargetIdx: 0,
      cachedMovementAction: 0,
      archetype: archetype || (presetName === "tank" ? "tank" : presetName === "scout" ? "melee" : "ranged"),
      threatTable: {},
    },
    targets, obstacles,
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
    step: 0, maxSteps: 100000, done: false,
    patterns: { aggression: 0, preferredRange: 0.5, evasion: 0, manaBurn: 0, directionEntropy: 0.5, directionCounts: Array(8).fill(0) },
  };
}

function targetPrioritySlots(sim: SimState): Target[] {
  return sim.targets.filter(t => t.alive).sort((a, b) => scoreTarget(sim, b) - scoreTarget(sim, a) || a.id - b.id).slice(0, TARGET_ACTIONS - 1);
}

// ═══════════════════════════════════════════════════════════════════
//  Sim Tick
// ═══════════════════════════════════════════════════════════════════
function tickSim(sim: SimState, action: [number, number, number], playerPos: [number, number], dt = PHYSICS_DT, playerActions: any = {}, decisionPulse = false): SimState {
  if (sim.done) return sim;
  const [mIdx, cIdx, tIdx] = action;
  const ag = sim.agent;
  const pl = sim.player;
  const half = sim.half;
  if (decisionPulse) sim.step++;

  // Decay threat table over time (C++ alignment: 5.0 points per second decay)
  if (!ag.threatTable) ag.threatTable = {};
  for (const id in ag.threatTable) {
    ag.threatTable[id] = Math.max(0, ag.threatTable[id] - 5.0 * dt);
    if (ag.threatTable[id] <= 0) {
      delete ag.threatTable[id];
    }
  }

  pl.pos = [playerPos[0], playerPos[1]];
  if (decisionPulse) {
    const priorityTargets = targetPrioritySlots(sim);
    if (tIdx < TARGET_ACTIONS - 1 && priorityTargets[tIdx]) {
      ag.activeTargetIdx = sim.targets.indexOf(priorityTargets[tIdx]);
    }
  }

  // Dynamic targeting: sync the player's position, facing, velocity, and HP to the target representing them
  const playerTarget = sim.targets.find(t => t.role === "player");
  if (playerTarget) {
    const px = playerTarget.pos[0], py = playerTarget.pos[1];
    playerTarget.vel = dt > 0 ? [(playerPos[0] - px) / dt, (playerPos[1] - py) / dt] : [0, 0];
    playerTarget.pos = [playerPos[0], playerPos[1]];
    playerTarget.facing = [pl.facing[0], pl.facing[1]];
    playerTarget.hp = pl.hp;
    playerTarget.alive = pl.hp > 0;
    playerTarget.commitment = playerActions.fireTarget ? 1 : Math.max(0, (playerTarget.commitment || 0) - dt / DT);
  }

  // Update player weapon
  const pw = pl.weapon;
  pw.cdRemain = Math.max(0, pw.cdRemain - dt);
  if (pw.isReloading) {
    pw.reloadRemain -= dt;
    if (pw.reloadRemain <= 0) { pw.isReloading = false; pw.ammo = pw.maxAmmo; }
  }

  if (playerActions.reload && !pw.isReloading && pw.ammo < pw.maxAmmo) {
    pw.isReloading = true;
    pw.reloadRemain = pw.reloadTime;
  }

  if (playerActions.fireTarget && pw.ammo > 0 && pw.cdRemain <= 0 && !pw.isReloading) {
    const target = playerActions.fireTarget;
    pw.ammo--;
    pw.cdRemain = pw.fireCd;
    const d = dist(pl.pos, target);
    if (d > 1) {
      pl.facing = norm([target[0] - pl.pos[0], target[1] - pl.pos[1]]);
    }
    spawnProjectile(sim, pl.pos, target,
      { projSpeed: pw.projSpeed, baseDmg: pw.baseDmg, range: pw.range, canArc: false },
      false, pl.atkStat, pl.critChance, pl.critMult, 0);
    const lastProj = sim.projectiles[sim.projectiles.length - 1];
    if (lastProj) { lastProj.isPlayer = true; lastProj.ownerId = 0; }
  }

  // AI Agent cooldowns
  ag.combatTime += dt;
  if (ag.lockRemain > 0) ag.lockRemain = Math.max(0, ag.lockRemain - dt);
  if (ag.dodgeCd > 0) ag.dodgeCd = Math.max(0, ag.dodgeCd - dt);
  if (ag.repositionCd > 0) ag.repositionCd = Math.max(0, ag.repositionCd - dt);
  if (ag.repositionRemain > 0) { ag.repositionRemain -= dt; if (ag.repositionRemain <= 0) ag.isRepositioning = false; }
  if (ag.isSwitching) { ag.switchRemain -= dt; if (ag.switchRemain <= 0) { ag.isSwitching = false; ag.activeWeapon = ag.switchTarget; } }
  if (ag.isWindingUp) { ag.windUpRemain -= dt; if (ag.windUpRemain <= 0) ag.isWindingUp = false; }
  ag.melee.cdRemain = Math.max(0, ag.melee.cdRemain - dt);
  for (const w of ag.weapons) {
    w.cdRemain = Math.max(0, w.cdRemain - dt);
    if (w.isReloading) { w.reloadRemain -= dt; if (w.reloadRemain <= 0) { w.isReloading = false; w.ammo = w.maxAmmo; } }
  }

  // Resolve pending fires
  if (ag.pendingFire && !ag.isWindingUp) {
    const pf = ag.pendingFire; ag.pendingFire = null;
    const slot = ag.weapons[pf.slotIdx];
    if (slot) {
      const d = dist(ag.pos, pf.targetPos);
      if (d <= slot.range && weaponHasPath(slot, ag.pos, pf.targetPos, sim.obstacles)) {
        spawnProjectile(sim, ag.pos, pf.targetPos, slot, true);
      }
    }
  }

  // Agent movement
  const isLocked = ag.lockRemain > 0;
  if (decisionPulse && !isLocked) ag.cachedMovementAction = mIdx;
  ag.isBlocking = !isLocked && cIdx === 6;
  if (decisionPulse && !isLocked && cIdx === 7 && ag.dodgeCd <= 0) {
    const requested = MOVE_DIRS[mIdx] || [0, 0];
    ag.dodgeDir = requested[0] || requested[1] ? [requested[0], requested[1]] : [-ag.facing[0], -ag.facing[1]];
    ag.dodgeStartPos = [ag.pos[0], ag.pos[1]];
    ag.dodgeEndPos = [ag.pos[0] + ag.dodgeDir[0] * ag.dodgeDistance, ag.pos[1] + ag.dodgeDir[1] * ag.dodgeDistance];
    ag.isDodging = true; ag.dodgeElapsed = 0; ag.dodgeRemain = ag.dodgeDuration;
    setLock(ag, ag.dodgeDuration, 3);
  } else if (decisionPulse && !isLocked && cIdx === 8 && ag.repositionCd <= 0) {
    const requested = MOVE_DIRS[mIdx] || [0, 0];
    ag.dodgeDir = requested[0] || requested[1] ? [requested[0], requested[1]] : [-ag.facing[0], -ag.facing[1]];
    ag.isRepositioning = true; ag.repositionRemain = ag.repositionDuration; ag.repositionCd = ag.repositionCooldown;
    setLock(ag, ag.repositionDuration, 7);
  }
  if (!isLocked && !ag.isDodging && !ag.isSwitching) {
    const dir = MOVE_DIRS[mIdx] || [0, 0];
    ag.vel = [dir[0] * ag.maxSpeed, dir[1] * ag.maxSpeed];
  }
  if (ag.isBlocking) ag.vel = [ag.vel[0] * ag.blockMoveMultiplier, ag.vel[1] * ag.blockMoveMultiplier];
  let dodgeOwnsMotion = false;
  if (ag.isDodging) {
    const oldPos: [number, number] = [ag.pos[0], ag.pos[1]];
    ag.dodgeElapsed = Math.min(ag.dodgeDuration, ag.dodgeElapsed + dt);
    const alpha = ag.dodgeElapsed / Math.max(ag.dodgeDuration, 1e-6);
    const eased = 1 - Math.pow(Math.max(0, 1 - alpha), ag.dodgeEaseExponent);
    const desired: [number, number] = [
      ag.dodgeStartPos[0] + (ag.dodgeEndPos[0] - ag.dodgeStartPos[0]) * eased,
      ag.dodgeStartPos[1] + (ag.dodgeEndPos[1] - ag.dodgeStartPos[1]) * eased,
    ];
    let hitT = 1;
    const sweepX = desired[0] - oldPos[0], sweepY = desired[1] - oldPos[1];
    for (const o of sim.obstacles) {
      const candidate = rayAABB(oldPos[0], oldPos[1], sweepX, sweepY, o.x, o.y, o.hw + AGENT_BODY_RADIUS, o.hh + AGENT_BODY_RADIUS);
      if (candidate !== null) hitT = Math.min(hitT, candidate);
    }
    ag.pos = hitT < 1
      ? [oldPos[0] + sweepX * Math.max(0, hitT - 0.0001), oldPos[1] + sweepY * Math.max(0, hitT - 0.0001)]
      : desired;
    ag.pos[0] = clamp(ag.pos[0], -half + AGENT_BODY_RADIUS, half - AGENT_BODY_RADIUS);
    ag.pos[1] = clamp(ag.pos[1], -half + AGENT_BODY_RADIUS, half - AGENT_BODY_RADIUS);
    pushOutAABB(ag.pos, sim.obstacles, AGENT_BODY_RADIUS);
    ag.vel = [(ag.pos[0] - oldPos[0]) / dt, (ag.pos[1] - oldPos[1]) / dt];
    ag.dodgeRemain = Math.max(0, ag.dodgeDuration - ag.dodgeElapsed);
    if (ag.dodgeRemain <= 0) { ag.isDodging = false; ag.dodgeCd = ag.dodgeCooldown; }
    dodgeOwnsMotion = true;
  } else if (ag.isRepositioning) {
    ag.vel = [ag.dodgeDir[0] * ag.maxSpeed * ag.repositionSpeedMultiplier, ag.dodgeDir[1] * ag.maxSpeed * ag.repositionSpeedMultiplier];
  }

  if (!dodgeOwnsMotion) {
    ag.pos[0] += ag.vel[0] * dt;
    ag.pos[1] += ag.vel[1] * dt;
  }
  ag.pos[0] = clamp(ag.pos[0], -half, half);
  ag.pos[1] = clamp(ag.pos[1], -half, half);
  pushOutAABB(ag.pos, sim.obstacles, 20);

  const target = sim.targets[clamp(ag.activeTargetIdx, 0, sim.targets.length - 1)];
  if (target?.alive) {
    const d = dist(ag.pos, target.pos);
    if (d > 1) ag.facing = norm([target.pos[0] - ag.pos[0], target.pos[1] - ag.pos[1]]);
  }

  // AI Combat moves
  if (decisionPulse && !isLocked && !ag.isSwitching && target?.alive) {
    const slot = ag.weapons[ag.activeWeapon];
    const d = dist(ag.pos, target.pos);
    const hasLOS = checkLOS(ag.pos, target.pos, sim.obstacles);

    if (cIdx === 1 && slot && slot.cdRemain <= 0 && slot.ammo > 0 && !slot.isReloading && !ag.isWindingUp) {
      slot.ammo--; slot.cdRemain = slot.fireCd;
      if (slot.windUp && slot.windUp > 0) {
        ag.isWindingUp = true; ag.windUpRemain = slot.windUp;
        setLock(ag, slot.windUp + slot.fireCd, 6);
        ag.pendingFire = { targetPos: [...target.pos], slotIdx: ag.activeWeapon };
      } else {
        setLock(ag, slot.fireCd * 0.5, 1);
        if (d <= slot.range && weaponHasPath(slot, ag.pos, target.pos, sim.obstacles)) {
          spawnProjectile(sim, ag.pos, target.pos, slot, true);
        }
      }
    } else if (cIdx === 2 && slot && !slot.isReloading && slot.ammo < slot.maxAmmo) {
      slot.isReloading = true; slot.reloadRemain = slot.reloadTime;
      setLock(ag, slot.reloadTime, 2);
      ag.isWindingUp = false; ag.pendingFire = null;
    } else if (cIdx === 3 && ag.weapons.length > 0 && ag.activeWeapon !== 0) {
      ag.isSwitching = true; ag.switchRemain = ag.switchTime; ag.switchTarget = 0;
      setLock(ag, ag.switchTime, 5);
    } else if (cIdx === 4 && ag.weapons.length > 1 && ag.activeWeapon !== 1) {
      ag.isSwitching = true; ag.switchRemain = ag.switchTime; ag.switchTarget = 1;
      setLock(ag, ag.switchTime, 5);
    } else if (cIdx === 5 && d <= ag.melee.range && ag.melee.cdRemain <= 0) {
      const { hpDamage, newBarrier } = computeDamage(ag.melee.damage, ag.atkStat, target.defence || 0, target.barrier || 0, ag.critChance, ag.critMult);
      target.barrier = newBarrier;
      target.hp -= hpDamage; if (target.hp <= 0) { target.hp = 0; target.alive = false; }
      ag.melee.cdRemain = ag.melee.cooldown;
      setLock(ag, ag.melee.cooldown, 4);
    }
  }

  // Hostile Red Targets AI
  for (const t of sim.targets) {
    if (!t.alive || t.role === "player") continue;
    t.atkCd -= dt;
    const toAgent = [ag.pos[0] - t.pos[0], ag.pos[1] - t.pos[1]];
    const dAgent = Math.sqrt(toAgent[0] ** 2 + toAgent[1] ** 2) || 1;
    t.facing = [toAgent[0] / dAgent, toAgent[1] / dAgent];

    const idealDist = 800;
    const approach = clamp((dAgent - idealDist) / idealDist, -0.5, 1);
    const perp = [-t.facing[1], t.facing[0]];
    const moveX = (t.facing[0] * approach * 0.6 + perp[0] * Math.sin(sim.step * 0.03 + t.id) * 0.3) * 250 * dt;
    const moveY = (t.facing[1] * approach * 0.6 + perp[1] * Math.sin(sim.step * 0.03 + t.id) * 0.3) * 250 * dt;
    t.pos[0] = clamp(t.pos[0] + moveX, -half, half);
    t.pos[1] = clamp(t.pos[1] + moveY, -half, half);
    pushOutAABB(t.pos, sim.obstacles, 15);

    if (t.atkCd <= 0 && dAgent < t.atkRange) {
      t.atkCd = 1.5 + Math.random();
      if (checkLOS(t.pos, ag.pos, sim.obstacles)) {
        const projData = { projSpeed: t.projSpeed, baseDmg: t.atkDmg, range: t.atkRange, canArc: false };
        spawnProjectile(sim, t.pos, ag.pos, projData, false, t.atkStat, t.critChance, t.critMult, t.id);
      }
    }
  }

  // Tick projectiles (highly optimized)
  const projSubsteps = Math.max(1, Math.round(dt * 60));
  const projDt = dt / projSubsteps;

  for (let sub = 0; sub < projSubsteps; sub++) {
    const activeProjectiles: Projectile[] = [];
    for (const p of sim.projectiles) {
      const oldX = p.pos[0], oldY = p.pos[1];

      if (p.canArc && p.arcStart && p.arcApex && p.arcEnd && p.arcFlightTime) {
        const previousProgress = Math.min(1, (p.arcElapsed || 0) / Math.max(p.arcFlightTime, 0.01));
        p.arcElapsed = (p.arcElapsed || 0) + projDt;
        const progress = Math.min(1, p.arcElapsed / Math.max(p.arcFlightTime, 0.01));
        const omt = 1 - progress;
        p.pos = [
          omt * omt * p.arcStart[0] + 2 * omt * progress * p.arcApex[0] + progress * progress * p.arcEnd[0],
          omt * omt * p.arcStart[1] + 2 * omt * progress * p.arcApex[1] + progress * progress * p.arcEnd[1],
        ];
        if (sub === 0) p.life -= dt;
        let blocked = false;
        const segmentX = p.pos[0] - oldX, segmentY = p.pos[1] - oldY;
        for (const o of sim.obstacles) {
          const hitT = rayAABB(oldX, oldY, segmentX, segmentY, o.x, o.y, o.hw, o.hh);
          if (hitT === null) continue;
          const arcT = previousProgress + (progress - previousProgress) * hitT;
          const projectileHeight = 2 * (1 - arcT) * arcT * (p.arcHeight || 400);
          if (projectileHeight <= o.height) { blocked = true; break; }
        }
        if (blocked || progress >= 1 || p.life <= 0) {
          applyArcImpact(sim, p);
          continue;
        }
        activeProjectiles.push(p);
        continue;
      }

      p.pos[0] += p.vel[0] * projDt;
      p.pos[1] += p.vel[1] * projDt;
      if (sub === 0) p.life -= dt;

      if (p.life <= 0 || Math.abs(p.pos[0]) > half || Math.abs(p.pos[1]) > half) {
        continue;
      }

      if (!p.canArc) {
        const dx = p.pos[0] - oldX, dy = p.pos[1] - oldY;
        let hitObs = false;
        for (const o of sim.obstacles) {
          if (rayAABB(oldX, oldY, dx, dy, o.x, o.y, o.hw, o.hh) !== null) {
            hitObs = true;
            break;
          }
        }
        if (hitObs) continue;
      }

      let hitEnt = false;
      const arenaScale = 460 / sim.arena;
      const radiusAgent = 14 / arenaScale;
      const radiusPlayer = 13 / arenaScale;
      const radiusTarget = 12 / arenaScale;

      if (p.isPlayer) {
        // Player's projectile: can damage AI Agent (ag) or Red Hostiles (t.role !== "player")
        if (!(ag.isDodging && ag.dodgeElapsed <= ag.dodgeInvulnDuration) && segPointDist(oldX, oldY, p.pos[0], p.pos[1], ag.pos[0], ag.pos[1]) < radiusAgent) {
          const defence = ag.defence + (ag.isBlocking ? ag.blockDefenceBonus : 0);
          const { hpDamage, newBarrier } = computeDamage(p.damage, p.atkStat, defence, ag.barrier, pl.critChance, pl.critMult);
          ag.barrier = newBarrier;
          ag.hp -= hpDamage;
          if (ag.hp <= 0) { ag.hp = 0; sim.done = true; }
          if (!ag.threatTable) ag.threatTable = {};
          ag.threatTable[0] = (ag.threatTable[0] || 0) + hpDamage;
          hitEnt = true;
        } else {
          for (const t of sim.targets) {
            if (!t.alive || t.role === "player") continue;
            if (segPointDist(oldX, oldY, p.pos[0], p.pos[1], t.pos[0], t.pos[1]) < radiusTarget) {
              const { hpDamage, newBarrier } = computeDamage(p.damage, p.atkStat, t.defence, t.barrier, pl.critChance, pl.critMult);
              t.barrier = newBarrier;
              t.hp -= hpDamage;
              if (t.hp <= 0) { t.hp = 0; t.alive = false; }
              hitEnt = true;
              break;
            }
          }
        }
      } else if (p.isAgent) {
        // AI Agent's projectile: can damage Player (t.role === "player") or Red Hostiles (t.role !== "player")
        for (const t of sim.targets) {
          if (!t.alive) continue;
          const targetRad = t.role === "player" ? radiusPlayer : radiusTarget;
          if (segPointDist(oldX, oldY, p.pos[0], p.pos[1], t.pos[0], t.pos[1]) < targetRad) {
            const { hpDamage, newBarrier } = computeDamage(p.damage, p.atkStat, t.defence, t.barrier, ag.critChance, ag.critMult);
            t.barrier = newBarrier;
            t.hp -= hpDamage;
            if (t.hp <= 0) {
              t.hp = 0; t.alive = false;
              if (t.role === "player") sim.done = true;
            }
            hitEnt = true;
            break;
          }
        }
      } else {
        // Red Hostile Robot's projectile: can damage AI Agent (ag) or Player (t.role === "player")
        if (!(ag.isDodging && ag.dodgeElapsed <= ag.dodgeInvulnDuration) && segPointDist(oldX, oldY, p.pos[0], p.pos[1], ag.pos[0], ag.pos[1]) < radiusAgent) {
          const defence = ag.defence + (ag.isBlocking ? ag.blockDefenceBonus : 0);
          const { hpDamage, newBarrier } = computeDamage(p.damage, p.atkStat, defence, ag.barrier);
          ag.barrier = newBarrier;
          ag.hp -= hpDamage;
          if (ag.hp <= 0) { ag.hp = 0; sim.done = true; }
          const oid = p.ownerId !== undefined ? p.ownerId : 1;
          if (!ag.threatTable) ag.threatTable = {};
          ag.threatTable[oid] = (ag.threatTable[oid] || 0) + hpDamage;
          hitEnt = true;
        } else {
          const playerTarget = sim.targets.find(t => t.role === "player");
          if (playerTarget && playerTarget.alive) {
            if (segPointDist(oldX, oldY, p.pos[0], p.pos[1], playerTarget.pos[0], playerTarget.pos[1]) < radiusPlayer) {
              const { hpDamage, newBarrier } = computeDamage(p.damage, p.atkStat, playerTarget.defence, playerTarget.barrier);
              playerTarget.barrier = newBarrier;
              playerTarget.hp -= hpDamage;
              if (playerTarget.hp <= 0) { playerTarget.hp = 0; playerTarget.alive = false; sim.done = true; }
              hitEnt = true;
            }
          }
        }
      }

      if (!hitEnt) {
        activeProjectiles.push(p);
      }
    }
    sim.projectiles = activeProjectiles;
  }

  const redHostiles = sim.targets.filter(t => t.role !== "player");
  if (redHostiles.length > 0) {
    if (redHostiles.every(t => !t.alive)) sim.done = true;
  } else {
    // 1v1 mode against AI: done if either agent is dead or player target is dead
    const playerTarget = sim.targets.find(t => t.role === "player");
    if (!playerTarget || !playerTarget.alive || ag.hp <= 0) sim.done = true;
  }

  if (sim.step >= sim.maxSteps) sim.done = true;

  // Sync back final player target state back to main player state
  const finalPlayerTarget = sim.targets.find(t => t.role === "player");
  if (finalPlayerTarget) {
    pl.hp = finalPlayerTarget.hp;
    if (pl.hp <= 0) {
      pl.hp = 0;
      sim.done = true;
    }
  }

  if (decisionPulse) updatePlayerPatterns(sim);

  return sim;
}

function updatePlayerPatterns(sim: SimState) {
  const alive = sim.targets.filter(t => t.alive);
  if (alive.length === 0) return;
  const p = sim.patterns;
  let aggression = 0, preferredRange = 0, manaBurn = 0;
  for (const t of alive) {
    if ((t.commitment ?? 0) > 0.01) aggression += 1;
    preferredRange += dist(t.pos, sim.agent.pos) / Math.max(sim.half * 2, 1);
    manaBurn += 1 - (t.mana ?? 0) / Math.max(t.maxMana ?? 1, 1);
    const speed = Math.hypot(t.vel[0], t.vel[1]);
    if (speed > 10) {
      const bin = Math.floor((Math.atan2(t.vel[1], t.vel[0]) + Math.PI) / (Math.PI / 4)) % 8;
      p.directionCounts[bin] += 1;
    }
  }
  const alpha = 0.05;
  p.aggression = p.aggression * (1 - alpha) + aggression / alive.length * alpha;
  p.preferredRange = p.preferredRange * (1 - alpha) + preferredRange / alive.length * alpha;
  p.manaBurn = p.manaBurn * (1 - alpha) + manaBurn / alive.length * alpha;
  p.directionCounts = p.directionCounts.map(v => v * 0.99);
  const total = p.directionCounts.reduce((a, b) => a + b, 0);
  if (total > 1e-6) {
    let entropy = 0;
    for (const count of p.directionCounts) {
      const probability = count / total;
      if (probability > 0.001) entropy -= probability * Math.log(probability);
    }
    p.directionEntropy = Math.min(1, entropy / Math.log(8));
  }
}

function setLock(ag: AgentState, duration: number, reason: number) {
  ag.lockRemain = duration; ag.lockDuration = duration; ag.lockReason = reason;
}

function spawnProjectile(sim: SimState, from: [number, number], to: [number, number], slot: any, isAgent: boolean, atkStat = 0, critChance = 0, critMult = 1.5, ownerId?: number) {
  const d = dist(from, to);
  const dir = norm([to[0] - from[0], to[1] - from[1]]);
  const spread = (Math.random() - 0.5) * 0.1;
  const cs = Math.cos(spread), sn = Math.sin(spread);
  const fd: [number, number] = [dir[0] * cs - dir[1] * sn, dir[0] * sn + dir[1] * cs];

  const arcHeight = 400;
  const halfDistance = d * 0.5;
  const arcLength = 2 * Math.sqrt(halfDistance * halfDistance + arcHeight * arcHeight);
  const arcFlightTime = arcLength / Math.max(slot.projSpeed, 500);
  sim.projectiles.push({
    pos: [from[0], from[1]], vel: [fd[0] * slot.projSpeed, fd[1] * slot.projSpeed],
    damage: slot.baseDmg, atkStat: isAgent ? sim.agent.atkStat : (atkStat || 0),
    critChance: isAgent ? sim.agent.critChance : critChance,
    critMult: isAgent ? sim.agent.critMult : critMult,
    isAgent, canArc: slot.canArc || false,
    life: slot.canArc ? arcFlightTime + 0.5 : Math.max(d / slot.projSpeed + 0.5, 1.0),
    ownerId, impactRadius: slot.canArc ? 150 : undefined,
    arcStart: slot.canArc ? [from[0], from[1]] : undefined,
    arcApex: slot.canArc ? [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5] : undefined,
    arcEnd: slot.canArc ? [to[0], to[1]] : undefined,
    arcHeight: slot.canArc ? arcHeight : undefined,
    arcFlightTime: slot.canArc ? arcFlightTime : undefined,
    arcElapsed: slot.canArc ? 0 : undefined,
  });
}

function applyArcImpact(sim: SimState, p: Projectile) {
  const radius = p.impactRadius || 0;
  if (radius <= 0) return;
  const actors = p.isAgent ? sim.targets : [sim.agent];
  for (const actor of actors) {
    if (actor.hp <= 0 || ("alive" in actor && !actor.alive)) continue;
    const impactDistance = dist(p.pos, actor.pos);
    if (impactDistance >= radius) continue;
    if (actor === sim.agent && actor.isDodging && actor.dodgeElapsed <= actor.dodgeInvulnDuration) continue;
    const falloff = 1 - impactDistance / radius * 0.5;
    const defence = actor.defence + (actor === sim.agent && actor.isBlocking ? actor.blockDefenceBonus : 0);
    const { hpDamage, newBarrier } = computeDamage(p.damage * falloff, p.atkStat, defence, actor.barrier, p.critChance, p.critMult);
    actor.barrier = newBarrier;
    actor.hp = Math.max(0, actor.hp - hpDamage);
    if (actor.hp <= 0) {
      if ("alive" in actor) actor.alive = false;
      if (actor === sim.agent || ("role" in actor && actor.role === "player")) sim.done = true;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Score Target (Match Python _score_target exactly)
// ═══════════════════════════════════════════════════════════════════
function scoreTarget(sim: SimState, t: Target): number {
  const ag = sim.agent;
  const d = dist(ag.pos, t.pos);
  const normDist = Math.min(d / 3000, 1.0);
  let playerWeight = 30;
  let lowHpWeight = 25;
  let threatWeight = 20;
  let distanceWeight = 10;
  let losWeight = 15;
  if (ag.archetype === "ranged") { losWeight *= 1.3; distanceWeight *= 0.8; }
  else if (ag.archetype === "melee") { distanceWeight *= 2.5; losWeight *= 0.5; lowHpWeight *= 1.5; playerWeight *= 1.2; }
  else if (ag.archetype === "healer") { threatWeight *= 0.3; playerWeight *= 0.5; distanceWeight *= 1.5; lowHpWeight *= 0.5; }
  else if (ag.archetype === "tank") { playerWeight *= 1.5; threatWeight *= 1.5; distanceWeight *= 1.3; lowHpWeight *= 0.5; }

  const pc = t.role === "player" ? playerWeight : 0;
  const lowHp = (1 - t.hp / Math.max(t.maxHp, 1)) * lowHpWeight;
  let threat = 0;
  if (ag.threatTable) {
    const rawThreat = ag.threatTable[t.id] || 0;
    let maxThreat = 0.01;
    for (const val of Object.values(ag.threatTable)) {
      if (val > maxThreat) maxThreat = val;
    }
    const normalizedThreat = rawThreat / maxThreat;
    threat = normalizedThreat * threatWeight;
  }
  const distanceScore = (1 - normDist) * distanceWeight;
  const los = checkLOS(ag.pos, t.pos, sim.obstacles) ? losWeight : 0;
  const active = sim.targets[ag.activeTargetIdx];
  const sticky = active?.id === t.id ? 8 : 0;
  const deterministicJitter = ((t.id * 1103515245 + sim.step * 12345) >>> 0) % 1000 / 100;
  return pc + lowHp + threat + distanceScore + los + sticky + deterministicJitter;
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
  fire_outside_optimal: -0.02,
  reload_when_empty: 0.02,
  wasted_shot: -0.0001,            // Was -0.03 — dramatically reduced
  passive_in_range: -0.08,         // NEW: per-step penalty for not firing when able
  target_low_hp_bonus: 3.0,        // NEW: one-time reward when target drops below 30%
  retarget_urgency: -0.06,         // NEW: per-step when selected target dead but others live
  arc_over_cover: 0.03,            // NEW: bonus for arcing over blocked LOS
  direct_fire_los: 0.02,           // NEW: bonus for direct weapon with clear LOS
  fire_valid_opportunity: 0.03,
  heavy_fire_intent: 0.02,
  fire_hit: 0.15,
  invalid_action: -0.1,
  death: -10.0,
};

interface RewardEngagementState {
  stepsSinceDamage: number;
  stepsSinceKill: number;
  seenLowHp: Set<number>;
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

  // ── 1. Damage dealt ──────────────────────────────────────────
  let totalDmgThisStep = 0;
  let killsThisStep = 0;
  const selectedPrev = prev.targets[agPrev.activeTargetIdx];
  let selectedDamage = 0;
  for (const tPrev of prev.targets) {
    const tCurr = curr.targets.find(t => t.id === tPrev.id);
    if (!tCurr) continue;
    const hpLoss = tPrev.hp - tCurr.hp;
    if (hpLoss > 0) {
      totalDmgThisStep += hpLoss;
      if (selectedPrev?.id === tPrev.id) selectedDamage += hpLoss;
    }
    if (tPrev.alive && !tCurr.alive) killsThisStep++;
  }

  const focusRatio = totalDmgThisStep > 1e-6 ? Math.min(selectedDamage / totalDmgThisStep, 1) : 0.5;
  const damageDealt = totalDmgThisStep * RW.damage_dealt * 100 * (0.5 + focusRatio);
  const killBonus = killsThisStep * RW.kill_reward;

  // ── 2. Engagement gate ───────────────────────────────────────
  // Binary gate: shaping rewards only active when damage dealt
  // within last 4 steps. Prevents chip-damage-every-7-steps exploit.
  if (totalDmgThisStep > 0) engState.stepsSinceDamage = 0;
  else engState.stepsSinceDamage++;
  if (killsThisStep > 0) engState.stepsSinceKill = 0;
  else engState.stepsSinceKill++;

  const engagement = engState.stepsSinceDamage <= 4 ? 1.0 : 0.0;

  // ── 3. Damage taken ──────────────────────────────────────────
  const hpLost = agPrev.hp - agCurr.hp;
  const damageTaken = hpLost > 0 ? hpLost * RW.damage_taken * 100 : 0;
  const death = agPrev.hp > 0 && agCurr.hp <= 0 ? RW.death : 0;

  // ── 4. Time penalty ──────────────────────────────────────────
  const timePenalty = RW.alive_per_step;

  // ── 5. Range rewards (engagement-gated) ──────────────────────
  let optimalRange = 0;
  let rangeClosing = 0;
  let outOfRange = 0;

  const activeTarget = curr.targets[agCurr.activeTargetIdx] || null;
  const rewardTarget = activeTarget?.alive ? activeTarget : (curr.targets.find(t => t.alive) || null);

  if (rewardTarget && stage >= 2) {
    const d = dist(agCurr.pos, rewardTarget.pos);
    const slot = agCurr.weapons[agCurr.activeWeapon];
    const optMin = slot?.optMin || 0;
    const optMax = slot?.optMax || (slot?.range || 1500);
    const range = slot?.range || 1500;

    if (d >= optMin && d <= optMax) {
      optimalRange = RW.in_optimal_range * engagement;  // GATED
    }

    if (d > range * 1.2) {
      const overshoot = d - range;
      outOfRange = RW.out_of_range_penalty * (0.3 + 0.7 * (overshoot / 800));
    }

    // Range closing (NOT gated — this IS the approach signal)
    const prevTarget = prev.targets.find(t => t.id === rewardTarget.id);
    if (prevTarget) {
      const prevD = dist(agPrev.pos, prevTarget.pos);
      const distClosed = prevD - d;
      if (distClosed > 0 && d > optMax) {
        const farBonus = 1 + Math.min(Math.max(d - range, 0) / 800, 2);
        rangeClosing = RW.range_closing * Math.min(distClosed / 80.0, 1.0) * farBonus;
      }
    }
  }

  // ── 6. Flanking (engagement-gated, stage 3+) ────────────────
  let flanking = 0;
  if (rewardTarget && stage >= 3) {
    const toT = norm([rewardTarget.pos[0] - agCurr.pos[0], rewardTarget.pos[1] - agCurr.pos[1]]);
    const facingDot = dot(rewardTarget.facing, toT);

    if (facingDot < -0.5) {
      flanking = RW.flanking_behind * engagement;  // GATED
    } else if (Math.abs(facingDot) < 0.5) {
      flanking = RW.flanking_side * engagement;    // GATED
    }
  }

  // ── 7. Inactivity penalty (stage 3+) ────────────────────────
  let inactivity = 0;
  if (stage >= 3) {
    const threshold = stage >= 6 ? 10 : (stage >= 5 ? 13 : 10);
    const postKillGrace = 35;
    const effective = engState.stepsSinceKill < postKillGrace
      ? Math.max(threshold, postKillGrace - engState.stepsSinceKill)
      : threshold;

    if (engState.stepsSinceDamage > effective) {
      const severity = Math.min((engState.stepsSinceDamage - effective) / 10, 3.0);
      inactivity = RW.damage_inactivity * severity;
    }
  }

  // ── 8. Weapon selection (engagement-gated, stage 4+) ────────
  let weaponSelection = 0;
  if (stage >= 4 && combatAction === 1 && rewardTarget) {
    const slot = agCurr.weapons[agCurr.activeWeapon];
    if (slot) {
      const d = dist(agCurr.pos, rewardTarget.pos);
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
  if (combatAction === 1) { // Fire
    const slot = agPrev.weapons[agPrev.activeWeapon];
    if (slot && rewardTarget) {
      const d = dist(agCurr.pos, rewardTarget.pos);
      const hasPath = weaponHasPath(slot, agCurr.pos, rewardTarget.pos, curr.obstacles);
      if (d > (slot.range || 1500) || !hasPath) {
        ammo = RW.wasted_shot;
      }
    }
  }
  if (combatAction === 2) { // Reload
    const slot = agPrev.weapons[agPrev.activeWeapon];
    if (slot && slot.ammo === 0) ammo = RW.reload_when_empty;
  }

  // ── 10. Current aggression/progression signals ──────────────
  let progression = 0;
  if (activeTarget && !activeTarget.alive && curr.targets.some(t => t.alive)) {
    progression += RW.retarget_urgency;
  }
  if (selectedPrev) {
    const selectedCurr = curr.targets.find(t => t.id === selectedPrev.id);
    if (selectedCurr && selectedDamage > 0 && selectedCurr.hp > 0 && selectedCurr.hp / selectedCurr.maxHp < 0.3 && !engState.seenLowHp.has(selectedCurr.id)) {
      progression += RW.target_low_hp_bonus;
      engState.seenLowHp.add(selectedCurr.id);
    }
  }

  let fireIntent = 0;
  let passive = 0;
  let invalid = 0;
  const prevSlot = agPrev.weapons[agPrev.activeWeapon];
  const prevTarget = prev.targets[agPrev.activeTargetIdx];
  const prevDistance = prevTarget?.alive ? dist(agPrev.pos, prevTarget.pos) : Infinity;
  const canFire = !!prevSlot && prevSlot.cdRemain <= 0 && prevSlot.ammo > 0 && !prevSlot.isReloading && agPrev.lockRemain <= 0;
  const viableFire = !!prevTarget?.alive && !!prevSlot && canFire && prevDistance <= prevSlot.range && weaponHasPath(prevSlot, agPrev.pos, prevTarget.pos, prev.obstacles);
  if (combatAction === 1 && !canFire) invalid = RW.invalid_action;
  if (combatAction === 2 && prevSlot && prevSlot.ammo >= prevSlot.maxAmmo) invalid = RW.invalid_action;
  if (combatAction === 1 && viableFire) {
    fireIntent += RW.fire_valid_opportunity;
    if ((prevSlot.windUp || 0) >= 0.3 || prevSlot.fireCd >= 0.8) fireIntent += RW.heavy_fire_intent;
    if (totalDmgThisStep > 0) fireIntent += RW.fire_hit;
    const clearLos = checkLOS(agPrev.pos, prevTarget!.pos, prev.obstacles);
    if (prevSlot.canArc && !clearLos) fireIntent += RW.arc_over_cover;
    if (!prevSlot.canArc && clearLos) fireIntent += RW.direct_fire_los;
  }
  const usefulBlock = combatAction === 6 && hpLost > 0;
  if ((combatAction === 0 || combatAction === 6) && !usefulBlock && viableFire) {
    passive = RW.passive_in_range;
  }

  // ── 11. Episode end ─────────────────────────────────────────
  let endBonus = 0;
  if (curr.done && !prev.done) {
    if (agCurr.hp <= 0) {
      endBonus = RW.episode_loss;
    } else {
      const aliveHostiles = curr.targets.filter(t => t.alive && t.role !== "player").length;
      if (aliveHostiles === 0) {
        const speedFrac = Math.max(0, 1.0 - stepCount / curr.maxSteps);
        endBonus = RW.episode_win * (1.0 + speedFrac);
      } else {
        // Timeout penalty scales with episode length
        const lengthScale = Math.max(1.0, stepCount / 200.0);
        endBonus = RW.episode_timeout * lengthScale + aliveHostiles * RW.surviving_target;
      }
    }
  }

  // ── Total ───────────────────────────────────────────────────
  const total = damageDealt + killBonus + damageTaken + death + timePenalty +
    optimalRange + rangeClosing + outOfRange +
    flanking + inactivity + weaponSelection + ammo + progression + fireIntent + passive + invalid + endBonus;

  return {
    reward: total,
    breakdown: {
      damageDealt,
      killBonus,
      damageTaken,
      death,
      timePenalty,
      optimalRange,
      rangeClosing,
      outOfRange,
      flanking,
      inactivity,
      weaponSelection,
      ammo,
      progression,
      fireIntent,
      passive,
      invalid,
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
  obs[idx++] = 0; // Stunned
  obs[idx++] = 0; // Slowed
  for (let i = 0; i < 6; i++) obs[idx++] = 0;
  const velDir = speed > 1 ? [ag.vel[0] / speed, ag.vel[1] / speed] : [0, 0];
  obs[idx++] = velDir[0];
  obs[idx++] = velDir[1];
  obs[idx++] = clamp(ag.combatTime / 120, 0, 1);

  // Calculate parabolic vertical height above ground during Dodge (C++ alignment)
  let height = 0;
  if (ag.isDodging && ag.dodgeDuration > 0) {
    const normTime = ag.dodgeRemain / ag.dodgeDuration; // 1.0 down to 0.0
    // Midpoint peak at 150 units up
    height = Math.max(0, 150 * 4 * normTime * (1 - normTime));
  }
  obs[idx++] = clamp(height / 500, 0, 1); // Height above ground (Self_14)

  obs[idx++] = ag.lockRemain > 0 ? 1 : 0;
  const lockProg = ag.lockRemain > 0 ? clamp(1 - ag.lockRemain / Math.max(ag.lockDuration, 0.01), 0, 1) : 0;
  obs[idx++] = lockProg;
  obs[idx++] = ag.lockRemain > 0 ? ag.lockReason / 7.0 : 0;
  obs[idx++] = ag.isDodging ? 1 : 0;
  obs[idx++] = ag.dodgeCd <= 0 ? 1 : 0;
  obs[idx++] = ag.isDodging && ag.dodgeElapsed <= ag.dodgeInvulnDuration ? 1 : 0;

  obs[idx++] = nSlots > 1 ? ag.activeWeapon / (nSlots - 1) : 0;
  obs[idx++] = slot ? slot.ammo / slot.maxAmmo : 0;
  obs[idx++] = (slot && slot.cdRemain <= 0 && slot.ammo > 0 && !slot.isReloading && ag.lockRemain <= 0) ? 1 : 0;
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

  const archetypes: AgentState["archetype"][] = ["ranged", "melee", "healer", "tank"];
  for (const archetype of archetypes) obs[idx++] = ag.archetype === archetype ? 1 : 0;
  const weaponRange = slot ? slot.range : 1000;
  obs[idx++] = clamp(weaponRange * 0.6 / 5000, 0, 1);
  const anyAmmo = ag.weapons.some(w => w.ammo > 0);
  obs[idx++] = anyAmmo ? 1 : 0;
  obs[idx++] = ag.melee.cdRemain <= 0 ? 1 : 0;

  const sortedTargets = sim.targets.filter(t => t.alive)
    .sort((a, b) => scoreTarget(sim, b) - scoreTarget(sim, a));
  const ct = sim.targets[ag.activeTargetIdx] || null;

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
    obs[idx++] = 1;
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
    obs[idx++] = ct.role === "player" ? 1 : 0; // is_player_controlled

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
    obs[idx++] = ct.characterType ?? 0.5;
    obs[idx++] = (ct.mana ?? 0) / Math.max(ct.maxMana ?? 1, 1);
    obs[idx++] = ct.commitment ?? 0;
    const gapCd = ct.gapCloserCd ?? 0;
    const gapReady = gapCd <= 0 && d <= (ct.gapCloserRange ?? 0);
    obs[idx++] = gapReady ? 1 : 0;
    obs[idx++] = ag.repositionCd <= 0 && !ag.isRepositioning ? 1 : 0;
  } else {
    idx += 23;
    obs[idx++] = ag.repositionCd <= 0 && !ag.isRepositioning ? 1 : 0;
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
      obs[base + 4] = t.hp / t.maxHp;
      obs[base + 5] = checkLOS(ag.pos, t.pos, sim.obstacles) ? 1 : 0;
      obs[base + 6] = t.role === "player" ? 1 : 0; // is_player_controlled
      const toAg = norm([ag.pos[0] - t.pos[0], ag.pos[1] - t.pos[1]]);
      obs[base + 7] = clamp(dot(t.facing, toAg), -1, 1); // raw dot [-1,1], matches C++
      obs[base + 8] = clamp(scoreTarget(sim, t) / 120.0, 0, 1); // score target
      let targetThreat = 0;
      if (ag.threatTable) {
        const rawThreat = ag.threatTable[t.id] || 0;
        let maxThreat = 0.01;
        for (const val of Object.values(ag.threatTable)) {
          if (val > maxThreat) maxThreat = val;
        }
        targetThreat = rawThreat / maxThreat;
      }
      obs[base + 9] = targetThreat;
      obs[base + 10] = clamp(t.vel[0] / 600, -1, 1);
      obs[base + 11] = clamp(t.vel[1] / 600, -1, 1);
      obs[base + 12] = clamp(dot(t.facing, toAg), 0, 1);
      obs[base + 13] = t.characterType ?? 0.5;
      obs[base + 14] = (t.mana ?? 0) / Math.max(t.maxMana ?? 1, 1);
      obs[base + 15] = t.commitment ?? 0;
      obs[base + 16] = (t.gapCloserCd ?? 0) <= 0 && d <= (t.gapCloserRange ?? 0) ? 1 : 0;
    }
  }
  idx = 142;
  idx = 187; // Allied robots skipped

  for (const ang of SPATIAL_ANGLES) {
    const rad = ang * Math.PI / 180;
    const dx = Math.cos(rad) * 1500, dy = Math.sin(rad) * 1500;
    let minT = 1.0;
    for (const o of sim.obstacles) {
      const t = rayAABB(ag.pos[0], ag.pos[1], dx, dy, o.x, o.y, o.hw + AGENT_BODY_RADIUS, o.hh + AGENT_BODY_RADIUS);
      if (t !== null && t < minT) minT = t;
    }
    const effectiveHalf = sim.half - AGENT_BODY_RADIUS;
    const wallTx = Math.abs(dx) > 1e-6 ? ((dx > 0 ? effectiveHalf : -effectiveHalf) - ag.pos[0]) / dx : Infinity;
    const wallTy = Math.abs(dy) > 1e-6 ? ((dy > 0 ? effectiveHalf : -effectiveHalf) - ag.pos[1]) / dy : Infinity;
    if (wallTx >= 0 && wallTx < minT) minT = wallTx;
    if (wallTy >= 0 && wallTy < minT) minT = wallTy;
    obs[idx++] = minT;
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
  obs[idx++] = 0;
  obs[idx++] = clamp(aliveH / 4, 0, 1);
  obs[idx++] = 0;
  obs[idx++] = aliveH > 0 ? sim.targets.filter(t => t.alive).reduce((s, t) => s + t.hp / t.maxHp, 0) / aliveH : 0;
  obs[idx++] = 0;
  obs[idx++] = aliveH > 0 ? 1 : 0;

  obs[idx++] = clamp(dist(ag.pos, ag.spawnPos || [0, 0]) / (ag.leashRange || 2000), 0, 1);

  // ── Extended Threat: Projectiles 2 and 3 (198-203) ──
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
  // Proj 2 (indices 198-200)
  if (threatProjs.length > 1) {
    obs[idx++] = threatProjs[1].dist;
    obs[idx++] = threatProjs[1].dir[0];
    obs[idx++] = threatProjs[1].dir[1];
  } else { idx += 3; }
  // Proj 3 (indices 201-203)
  if (threatProjs.length > 2) {
    obs[idx++] = threatProjs[2].dist;
    obs[idx++] = threatProjs[2].dir[0];
    obs[idx++] = threatProjs[2].dir[1];
  } else { idx += 3; }
  // Threat count (index 204)
  obs[idx++] = clamp(threatProjs.length / 5, 0, 1);

  // ── Can Hit Target per Weapon (205-208) ──
  const tgt = sim.targets[ag.activeTargetIdx];
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
      const hasPath = tgt && tgt.alive ? weaponHasPath(w, ag.pos, tgt.pos, sim.obstacles) : false;
      const canHit = w.ammo > 0 && inRange && hasPath && !w.isReloading;
      obs[idx++] = canHit ? 1 : 0;
    } else { idx++; }
  }

  // ── Total Ammo Fraction (209) ──
  const totalAmmo = ag.weapons.length > 0
    ? ag.weapons.reduce((s, w) => s + w.ammo / w.maxAmmo, 0) / ag.weapons.length
    : 0;
  obs[idx++] = clamp(totalAmmo, 0, 1);

  // ── Targets Killed Fraction (210) ──
  const totalHostiles = sim.targets.length;
  const killedHostiles = sim.targets.filter(t => !t.alive).length;
  obs[idx++] = totalHostiles > 0 ? killedHostiles / totalHostiles : 0;

  // ── Arc Clearance per Weapon (211-214) ──
  for (let wi = 0; wi < 4; wi++) {
    if (wi < ag.weapons.length) {
      const w = ag.weapons[wi];
      if (w.canArc) {
        const maxArc = w.maxArcHeight || 0;
        obs[idx++] = maxArc <= 0 ? 1.0 : clamp(maxArc / 3000, 0, 1);
      } else { idx++; }
    } else { idx++; }
  }

  obs[idx++] = sim.patterns.aggression;
  obs[idx++] = sim.patterns.evasion;
  obs[idx++] = sim.patterns.directionEntropy;
  obs[idx++] = sim.patterns.preferredRange;
  obs[idx++] = sim.patterns.manaBurn;

  if (idx !== OBS_SIZE) throw new Error(`Observation size mismatch: wrote ${idx}, expected ${OBS_SIZE}`);

  return obs;
}

// ═══════════════════════════════════════════════════════════════════
//  Build Action Mask
// ═══════════════════════════════════════════════════════════════════
function buildActionMask(sim: SimState) {
  const ag = sim.agent;
  const slot = ag.weapons[ag.activeWeapon];
  const isLocked = ag.lockRemain > 0;
  const m = new Array(MOVEMENT_ACTIONS).fill(!isLocked);
  if (isLocked) m[clamp(ag.cachedMovementAction, 0, MOVEMENT_ACTIONS - 1)] = true;

  const c = new Array(COMBAT_ACTIONS).fill(false);
  c[0] = true;
  if (!isLocked && !ag.isSwitching) {
    if (slot && slot.cdRemain <= 0 && slot.ammo > 0 && !slot.isReloading && !ag.isWindingUp) c[1] = true;
    if (slot && !slot.isReloading && slot.ammo < slot.maxAmmo) c[2] = true;
    if (ag.weapons.length > 0 && ag.activeWeapon !== 0) c[3] = true;
    if (ag.weapons.length > 1 && ag.activeWeapon !== 1) c[4] = true;
    if (ag.melee.cdRemain <= 0 && !ag.isDodging) c[5] = true;
    c[6] = true;
    if (!ag.isDodging && ag.dodgeCd <= 0) c[7] = true;
    if (!ag.isRepositioning && ag.repositionCd <= 0) c[8] = true;
  }

  const t = new Array(TARGET_ACTIONS).fill(false);
  const alive = sim.targets.filter(x => x.alive);
  for (let i = 0; i < Math.min(alive.length, 4); i++) t[i] = true;
  t[4] = true;

  return { m, c, t, skipInference: isLocked };
}

// ═══════════════════════════════════════════════════════════════════
//  ONNX Inference
// ═══════════════════════════════════════════════════════════════════
type TierKey = keyof typeof TIER_PROFILES;
interface ModelInfo { session: ort.InferenceSession; frameStack: number; hiddenSize: number; tier: TierKey; }
const TIER_HIDDEN: Record<TierKey, number> = { micro: 32, small: 48, medium: 48, large: 64 };

async function loadOnnxModel(file: File): Promise<ModelInfo> {
  const buf = await file.arrayBuffer();
  const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  const lowerName = file.name.toLowerCase();
  const tier = (["micro", "small", "medium", "large"] as TierKey[]).find(t => lowerName.includes(t)) || "medium";
  const metadata = session.inputMetadata as any;
  const obsIndex = session.inputNames.indexOf("observation");
  const hiddenIndex = session.inputNames.indexOf("hidden_in");
  const obsDims = metadata?.[obsIndex]?.shape || [];
  const hiddenDims = metadata?.[hiddenIndex]?.shape || [];
  const inputSize = Number(obsDims[obsDims.length - 1]);
  const hiddenFromModel = Number(hiddenDims[hiddenDims.length - 1]);
  const frameStack = Number.isFinite(inputSize) && inputSize % OBS_SIZE === 0 ? inputSize / OBS_SIZE : 3;
  const hiddenSize = Number.isFinite(hiddenFromModel) && hiddenFromModel > 0 ? hiddenFromModel : TIER_HIDDEN[tier];
  if (!session.inputNames.includes("observation") || !session.inputNames.includes("hidden_in")) {
    throw new Error(`Unsupported model inputs: ${session.inputNames.join(", ")}`);
  }
  return { session, frameStack, hiddenSize, tier };
}

async function runInference(model: ModelInfo, obsBuffer: Float32Array, hidden: Float32Array) {
  const tensor = new ort.Tensor("float32", obsBuffer, [1, obsBuffer.length]);
  const hiddenTensor = new ort.Tensor("float32", hidden, [1, 1, model.hiddenSize]);
  const results = await model.session.run({ observation: tensor, hidden_in: hiddenTensor });
  return {
    m: Array.from(results.movement_logits.data as Float32Array),
    c: Array.from(results.combat_logits.data as Float32Array),
    t: Array.from(results.target_logits.data as Float32Array),
    hidden: new Float32Array(results.hidden_out.data as Float32Array),
  };
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
  fs.buf[fs.idx % fs.frameStack] = new Float32Array(obs);
  fs.idx++;
  if (fs.idx >= fs.frameStack) fs.filled = true;
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
    if (!t.alive || t.role === "player") return;
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
  const frameStackBufRef = useRef<FrameStack | null>(null);
  const hiddenStateRef = useRef<Float32Array | null>(null);
  const inferencePendingRef = useRef(false);
  const pendingDecisionRef = useRef<{ action: [number, number, number]; observation: Float32Array } | null>(null);
  const rewardSnapshotRef = useRef<SimState | null>(null);
  const rewardActionRef = useRef<[number, number, number]>([0, 0, TARGET_ACTIONS - 1]);
  const rewardObservationRef = useRef<Float32Array | null>(null);
  const playerActionsRef = useRef<{ fireTarget: [number, number] | null; reload: boolean }>({ fireTarget: null, reload: false });

  // Reactive state for HUD/Settings
  const [visualSim, setVisualSim] = useState<SimState | null>(null);
  const [lastActionState, setLastActionState] = useState<[number, number, number]>([0, 0, 4]);
  const [isPaused, setIsPaused] = useState(false);
  const [status, setStatus] = useState("No model loaded — AI uses scripted behavior");
  const [preset, setPreset] = useState<"heavy" | "scout" | "sniper" | "tank">("heavy");
  const [archetype, setArchetype] = useState<AgentState["archetype"]>("ranged");
  const [numTargets, setNumTargets] = useState(2);
  const [numObs, setNumObs] = useState(4);
  const [arenaSize, setArenaSize] = useState(3500);
  const [overlays, setOverlays] = useState({ facing: true, range: true, los: true });

  // C++ Alignment States for Registry/Decision / Sampling Modes
  const [selectedTier, setSelectedTier] = useState<TierKey>("medium");
  const [sampleStrategy, setSampleStrategy] = useState<"greedy" | "stochastic">("greedy");
  const [temperature, setTemperature] = useState<number>(1.0);

  // C++ Style Data Recording State (Part 2 Align)
  const [isRecording, setIsRecording] = useState(false);
  const [recordedStepCount, setRecordedStepCount] = useState(0);
  const recordedStepsRef = useRef<any[]>([]);
  const isRecordingRef = useRef(false);
  const encounterIdRef = useRef(1);

  const engagementStateRef = useRef<RewardEngagementState>({ stepsSinceDamage: 0, stepsSinceKill: 999, seenLowHp: new Set() });

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
  const modelInfoRef = useRef<ModelInfo | null>(null);

  const selectedTierRef = useRef<TierKey>("medium");
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
  const resetSim = useCallback((selPreset = preset, targetsCount = numTargets, obsCount = numObs, selArenaSize = arenaSize, selArchetype = archetype) => {
    encounterIdRef.current += 1;
    const s = createSim(selPreset, selArenaSize, targetsCount, obsCount, selArchetype);
    simRef.current = s;
    playerPosRef.current = [selArenaSize * 0.2, selArenaSize * 0.12];
    if (modelInfoRef.current) {
      frameStackBufRef.current = createFrameStack(modelInfoRef.current.frameStack);
      hiddenStateRef.current = new Float32Array(modelInfoRef.current.hiddenSize);
    }
    inferencePendingRef.current = false;
    pendingDecisionRef.current = null;
    rewardSnapshotRef.current = structuredClone(s) as SimState;
    rewardActionRef.current = [0, 0, TARGET_ACTIONS - 1];
    rewardObservationRef.current = null;
    lastActionRef.current = [0, 0, TARGET_ACTIONS - 1];
    setLastActionState([0, 0, TARGET_ACTIONS - 1]);

    // Reset reward lists and accumulators
    rewardHistoryRef.current = [];
    setRewardHistory([]);
    engagementStateRef.current = { stepsSinceDamage: 0, stepsSinceKill: 999, seenLowHp: new Set() };
    accumRewardRef.current = 0;
    accumBreakdownRef.current = {
      damageDealt: 0,
      timePenalty: 0,
      hpLossPenalty: 0,
      meleeRangeBonus: 0,
      meleeDistancePenalty: 0,
      optimalRangeBonus: 0,
      outOfRangePenalty: 0,
      behindCoverBonus: 0,
      flankingBonus: 0,
      weaponSwitch: 0,
      endBonus: 0,
    };
    aiTickTimerRef.current = 0;

    setVisualSim(s);
  }, [preset, numTargets, numObs, arenaSize, archetype]);

  const downloadRecordedCSV = () => {
    if (recordedStepsRef.current.length === 0) return;

    // 258 fields total: 5 metadata + 249 observation vector + 3 action vector + 1 reward
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

  // Load and configure simulation on settings update
  useEffect(() => {
    resetSim(preset, numTargets, numObs, arenaSize, archetype);
  }, [preset, numTargets, numObs, arenaSize, archetype]);

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

  // Decoupled Game Loop running via requestAnimationFrame
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();
    let physicsAccumulator = 0;
    let hudAccumulator = 0;

    const publishVisual = (sim: SimState) => {
      setVisualSim({
        ...sim,
        agent: { ...sim.agent, weapons: sim.agent.weapons.map(w => ({ ...w })) },
        targets: sim.targets.map(t => ({ ...t })),
        obstacles: sim.obstacles.map(o => ({ ...o })),
        projectiles: sim.projectiles.map(p => ({ ...p })),
        player: { ...sim.player, weapon: { ...sim.player.weapon } },
      });
    };

    const finishRewardWindow = (sim: SimState) => {
      const previous = rewardSnapshotRef.current;
      if (!previous || sim.step <= 0) return;
      const action = rewardActionRef.current;
      const stepRewardData = calculateFrameReward(previous, sim, action, stageRef.current, sim.step, DT, engagementStateRef.current);
      const actionDesc = `Move[${action[0]}] + ${COMBAT_NAMES[action[1]] || "None"} (${action[2] === TARGET_ACTIONS - 1 ? "Keep" : `T${action[2]}`})`;
      const newLog: RewardStepLog = {
        step: sim.step,
        action,
        actionDesc,
        reward: stepRewardData.reward,
        breakdown: stepRewardData.breakdown,
        cumReward: (rewardHistoryRef.current[rewardHistoryRef.current.length - 1]?.cumReward || 0) + stepRewardData.reward,
      };
      rewardHistoryRef.current.push(newLog);
      if (rewardHistoryRef.current.length > 300) rewardHistoryRef.current.shift();
      setRewardHistory([...rewardHistoryRef.current]);

      if (isRecordingRef.current) {
        recordedStepsRef.current.push({
          encounterId: encounterIdRef.current,
          enemyName: TIER_PROFILES[selectedTierRef.current].name,
          archetype: sim.agent.archetype.toUpperCase(),
          frame: sim.step,
          combatTime: sim.agent.combatTime,
          observation: Array.from(rewardObservationRef.current || new Float32Array(OBS_SIZE)),
          action,
          reward: stepRewardData.reward,
        });
        setRecordedStepCount(recordedStepsRef.current.length);
      }
    };

    const requestPolicyDecision = (sim: SimState, playerPos: [number, number]) => {
      if (pendingDecisionRef.current) return;
      const mask = buildActionMask(sim);
      setCurrentMasks(mask);
      const obs = buildObservation(sim, playerPos, prevTargetVelRef.current, DT);
      sim.targets.forEach(t => {
        if (t.alive) prevTargetVelRef.current[t.id] = [t.vel[0], t.vel[1]];
      });
      currentObsRef.current = obs;
      setCurrentObs(new Float32Array(obs));

      if (mask.skipInference) {
        pendingDecisionRef.current = {
          action: [sim.agent.cachedMovementAction, 0, TARGET_ACTIONS - 1],
          observation: obs,
        };
        return;
      }

      const model = modelInfoRef.current;
      const frameStack = frameStackBufRef.current;
      if (!model || !frameStack) {
        pendingDecisionRef.current = { action: scriptedAI(sim, playerPos), observation: obs };
        return;
      }
      if (inferencePendingRef.current) return;

      pushFrame(frameStack, obs);
      const stacked = getStacked(frameStack);
      const hidden = hiddenStateRef.current || new Float32Array(model.hiddenSize);
      const requestEncounter = encounterIdRef.current;
      inferencePendingRef.current = true;
      runInference(model, stacked, hidden).then(logits => {
        if (requestEncounter !== encounterIdRef.current || modelInfoRef.current !== model) return;
        hiddenStateRef.current = logits.hidden;
        setCurrentLogits(logits);
        const action: [number, number, number] = sampleStrategyRef.current === "stochastic"
          ? [softmaxSample(logits.m, mask.m, tempRef.current), softmaxSample(logits.c, mask.c, tempRef.current), softmaxSample(logits.t, mask.t, tempRef.current)]
          : [argmaxMasked(logits.m, mask.m), argmaxMasked(logits.c, mask.c), argmaxMasked(logits.t, mask.t)];
        pendingDecisionRef.current = { action, observation: obs };
      }).catch(err => console.error("ONNX model run error:", err))
        .finally(() => { inferencePendingRef.current = false; });
    };

    const loop = (time: number) => {
      animId = requestAnimationFrame(loop);

      if (pausedRef.current) {
        lastTime = time;
        physicsAccumulator = 0;
        return;
      }

      if (!simRef.current) {
        lastTime = time;
        return;
      }

      let frameDt = (time - lastTime) / 1000;
      if (frameDt > 0.1) frameDt = 0.1;
      if (frameDt <= 0) frameDt = PHYSICS_DT;
      lastTime = time;
      physicsAccumulator = Math.min(physicsAccumulator + frameDt, PHYSICS_DT * 5);
      hudAccumulator += frameDt;

      const pp = playerPosRef.current;
      const keys = keysRef.current;

      while (physicsAccumulator >= PHYSICS_DT) {
        const currentSim = simRef.current;
        if (!currentSim || currentSim.done) break;

        let decisionPulse = false;
        const pending = pendingDecisionRef.current;
        if (pending) {
          finishRewardWindow(currentSim);
          rewardSnapshotRef.current = structuredClone(currentSim) as SimState;
          rewardActionRef.current = pending.action;
          rewardObservationRef.current = new Float32Array(pending.observation);
          lastActionRef.current = pending.action;
          setLastActionState(pending.action);
          pendingDecisionRef.current = null;
          decisionPulse = true;
        }

        const playerSpeed = 350;
        let dx = 0, dy = 0;
        if (keys.w || keys.arrowup) dy -= 1;
        if (keys.s || keys.arrowdown) dy += 1;
        if (keys.a || keys.arrowleft) dx -= 1;
        if (keys.d || keys.arrowright) dx += 1;
        if (dx !== 0 || dy !== 0) {
          const length = Math.sqrt(dx * dx + dy * dy);
          pp[0] += dx / length * playerSpeed * PHYSICS_DT;
          pp[1] += dy / length * playerSpeed * PHYSICS_DT;
        }
        pp[0] = clamp(pp[0], -currentSim.half, currentSim.half);
        pp[1] = clamp(pp[1], -currentSim.half, currentSim.half);
        pushOutAABB(pp, currentSim.obstacles, 15);
        playerPosRef.current = [pp[0], pp[1]];

        const playerActions = { ...playerActionsRef.current };
        playerActionsRef.current = { fireTarget: null, reload: false };
        const nextSim = tickSim(currentSim, lastActionRef.current, [pp[0], pp[1]], PHYSICS_DT, playerActions, decisionPulse);
        simRef.current = nextSim;

        aiTickTimerRef.current += PHYSICS_DT;
        if (aiTickTimerRef.current >= DT && !nextSim.done) {
          aiTickTimerRef.current = Math.max(0, aiTickTimerRef.current - DT);
          requestPolicyDecision(nextSim, [pp[0], pp[1]]);
        }
        if (nextSim.done && rewardSnapshotRef.current) {
          finishRewardWindow(nextSim);
          rewardSnapshotRef.current = null;
        }
        physicsAccumulator -= PHYSICS_DT;
      }

      const nextSim = simRef.current;
      if (!nextSim) return;

      // 5. Direct draw to canvas
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, 460, 460);
          render(ctx, 460, 460, nextSim, [pp[0], pp[1]], overlaysRef.current);
        }
      }

      // HUD panels are intentionally throttled; canvas rendering stays at RAF speed.
      if (hudAccumulator >= 0.1 || nextSim.done) {
        hudAccumulator = 0;
        publishVisual(nextSim);
      }
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Support Model upload with automatic C++ Model Tier profile detection
  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    setStatus("Loading ONNX Model...");
    try {
      const info = await loadOnnxModel(file);
      modelInfoRef.current = info;
      frameStackBufRef.current = createFrameStack(info.frameStack);
      hiddenStateRef.current = new Float32Array(info.hiddenSize);
      setSelectedTier(info.tier);
      setStatus(`Loaded ${info.tier.toUpperCase()} ONNX model (0.2s decisions, ${info.frameStack} frames, GRU ${info.hiddenSize})`);
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
    const maxEpSteps = 400;

    for (let ep = 0; ep < n; ep++) {
      const sim = createSim(preset, numTargets, numObs, arenaSize, archetype);
      const fs = createFrameStack(model.frameStack);
      let hidden = new Float32Array(model.hiddenSize);
      const prevVelMap: Record<number, [number, number]> = {};
      const engState: RewardEngagementState = { stepsSinceDamage: 0, stepsSinceKill: 999, seenLowHp: new Set() };
      let cumReward = 0;
      let kills = 0;
      let epStep = 0;

      // Fixed player position for consistency
      const fixedPlayerPos: [number, number] = [arenaSize * 0.2, arenaSize * 0.12];

      while (!sim.done && epStep < maxEpSteps) {
        const mask = buildActionMask(sim);
        const obs = buildObservation(sim, fixedPlayerPos, prevVelMap, DT);
        let action: [number, number, number];
        if (mask.skipInference) {
          action = [sim.agent.cachedMovementAction, 0, TARGET_ACTIONS - 1];
        } else {
          pushFrame(fs, obs);
          const logits = await runInference(model, getStacked(fs), hidden);
          hidden = logits.hidden;
          action = [argmaxMasked(logits.m, mask.m), argmaxMasked(logits.c, mask.c), argmaxMasked(logits.t, mask.t)];
        }

        // Track kills before tick
        const prevAlive = sim.targets.filter(t => t.alive && t.role !== "player").length;

        const prevSim = structuredClone(sim) as SimState;
        tickSim(sim, action, fixedPlayerPos, DT, {}, true);

        const currAlive = sim.targets.filter(t => t.alive && t.role !== "player").length;
        const newKills = prevAlive - currAlive;
        kills += newKills;

        // Simple reward calc for batch
        const stepReward = calculateFrameReward(prevSim as SimState, sim, action, stage, sim.step, DT, engState);
        cumReward += stepReward.reward;

        sim.targets.forEach(t => { if (t.alive) prevVelMap[t.id] = [t.vel[0], t.vel[1]]; });
        epStep++;
      }

      const hostiles = sim.targets.filter(t => t.role !== "player");
      const win = hostiles.length > 0 ? hostiles.every(t => !t.alive) : sim.agent.hp > 0;
      episodes.push({ win, reward: cumReward, kills, length: epStep });
    }

    return { episodes } as BatchResults;
  }, [preset, numTargets, numObs, arenaSize, archetype, stage]);

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
                const hostiles = visualSim.targets.filter(t => t.role !== "player");
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
                {visualSim.done && <span style={{ color: visualSim.targets.filter(t => t.role !== "player").every(t => !t.alive) ? "#7ee787" : "#f85149", fontWeight: 700 }}>
                  {visualSim.targets.filter(t => t.role !== "player").every(t => !t.alive) ? "AI Wins" : visualSim.agent.hp <= 0 ? "You Win" : "Timeout"}
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
                        <select value={preset} onChange={e => setPreset(e.target.value as any)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}><option value="heavy">Heavy</option><option value="scout">Scout</option><option value="sniper">Sniper</option><option value="tank">Tank</option></select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Archetype
                        <select value={archetype} onChange={e => setArchetype(e.target.value as AgentState["archetype"])} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}><option value="ranged">Ranged</option><option value="melee">Melee</option><option value="healer">Healer</option><option value="tank">Tank</option></select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Arena Size
                        <select value={arenaSize} onChange={e => setArenaSize(+e.target.value)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[2000, 2500, 3000, 3500, 4000, 5000].map(s => <option key={s} value={s}>{s}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Stage
                        <select value={stage} onChange={e => setStage(+e.target.value)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[1, 2, 3, 4, 5, 6, 7].map(n => <option key={n} value={n}>{n}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Targets
                        <select value={numTargets} onChange={e => setNumTargets(+e.target.value)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[0, 1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Obstacles
                        <select value={numObs} onChange={e => setNumObs(+e.target.value)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}>{[0, 2, 4, 6, 8, 12, 16].map(n => <option key={n} value={n}>{n}</option>)}</select>
                      </label>
                      <label style={{ color: "#8b949e", fontSize: 11 }}>Tier Profile
                        <select value={selectedTier} onChange={e => setSelectedTier(e.target.value as TierKey)} style={{ ...sel, display: "block", width: "100%", marginTop: 2, marginLeft: 0 }}><option value="micro">Micro (5Hz)</option><option value="small">Small (5Hz)</option><option value="medium">Medium (5Hz)</option><option value="large">Large (5Hz)</option></select>
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
                      {visualSim.targets.filter(t => t.role !== "player").map(t => (
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
                          numTargets={visualSim.targets.filter(t => t.role !== "player").length}
                          kills={visualSim.targets.filter(t => t.role !== "player" && !t.alive).length}
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
  for (let i = 0; i < 9; i++) {
    const dd = dot(MOVE_DIRS[i] as [number, number], norm(moveDir));
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
