import React, { useState, useRef, useEffect } from "react";

const OBS_GROUPS = [
  { name: "Self State", start: 0, end: 21, color: "#58a6ff", labels: [
    "HP%", "Defence", "Speed%", "Stunned", "Slowed", "Buff0", "Buff1", "Buff2", "Buff3", "Buff4", "Buff5",
    "VelDirX", "VelDirY", "CombatTime", "Height", "IsLocked", "LockProgress", "LockReason",
    "IsDodging", "DodgeReady", "IsInvuln",
  ]},
  { name: "Weapon State", start: 21, end: 43, color: "#ffa657", labels: [
    "ActiveIdx", "Ammo%", "CanFire", "IsReloading", "ReloadProg", "Range", "FireCd",
    "WindUp", "CanArc", "IsRanged",
    "Alt0Ammo", "Alt0Range", "Alt0Reload", "Alt0Arc",
    "Alt1Ammo", "Alt1Range", "Alt1Reload", "Alt1Arc",
    "Alt2Ammo", "Alt2Range", "Alt2Reload", "Alt2Arc",
  ]},
  { name: "Archetype", start: 43, end: 50, color: "#f0883e", labels: [
    "Arch0", "Arch1", "Arch2", "Arch3", "OptRange", "AnyAmmo", "MeleeReady",
  ]},
  { name: "Primary Target", start: 50, end: 70, color: "#f85149", labels: [
    "RelX", "RelY", "Dist", "HP%", "InRange", "HasLOS", "InSightCone", "SelfFacing",
    "TargetFacingMe", "VelX", "VelY", "AccelX", "AccelY",
    "AngSize", "IsPlayer", "BehindLowCover", "CoverHeight", "InMelee", "ClosingRate", "Pad",
  ]},
  { name: "Hostile 0", start: 70, end: 83, color: "#da3633" },
  { name: "Hostile 1", start: 83, end: 96, color: "#da3633" },
  { name: "Hostile 2", start: 96, end: 109, color: "#da3633" },
  { name: "Hostile 3", start: 109, end: 122, color: "#da3633" },
  { name: "Ally 0", start: 122, end: 134, color: "#3fb950" },
  { name: "Ally 1", start: 134, end: 146, color: "#3fb950" },
  { name: "Ally 2", start: 146, end: 158, color: "#3fb950" },
  { name: "Spatial Ring", start: 158, end: 166, color: "#79c0ff" },
  { name: "Cover Height", start: 166, end: 174, color: "#a5d6ff" },
  { name: "Threat Sense", start: 174, end: 182, color: "#ff7b72" },
  { name: "Navmesh", start: 182, end: 191, color: "#d2a8ff" },
  { name: "Group Summary", start: 191, end: 197, color: "#8b949e" },
  { name: "Spawn/Leash", start: 197, end: 198, color: "#8b949e" },
  { name: "Ext Threat", start: 198, end: 205, color: "#ff7b72", labels: [
    "Proj2Dist", "Proj2DirX", "Proj2DirY", "Proj3Dist", "Proj3DirX", "Proj3DirY", "ThreatCount",
  ]},
  { name: "Can Hit", start: 205, end: 209, color: "#ffa657", labels: ["Slot0", "Slot1", "Slot2", "Slot3"] },
  { name: "Ammo/Kills", start: 209, end: 211, color: "#8b949e", labels: ["TotalAmmo", "KillFrac"] },
  { name: "Arc Clearance", start: 211, end: 215, color: "#d2a8ff", labels: ["Slot0", "Slot1", "Slot2", "Slot3"] },
];

const HOSTILE_LABELS = ["Alive", "RelX", "RelY", "Dist", "HP%", "HasLOS", "IsPlayer", "FacingMe", "Score", "Threat", "VelX", "VelY", "FacingDot"];
const ALLY_LABELS = ["Alive", "RelX", "RelY", "Dist", "HP%", "Ammo", "InCombat", "Dodging", "Archetype", "VelX", "VelY", "TgtSlot"];

interface Props {
  obs: Float32Array | null;
}

function ObsValue({ value, prevValue, label, idx }: { value: number; prevValue: number; label?: string; idx: number }) {
  const changed = Math.abs(value - prevValue) > 0.001;
  const direction = value > prevValue ? "up" : value < prevValue ? "down" : "same";

  return (
    <div
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        width: 44,
        padding: "1px 2px",
        borderRadius: 3,
        background: changed
          ? direction === "up"
            ? "rgba(126,231,135,0.12)"
            : "rgba(255,123,114,0.12)"
          : "transparent",
        transition: "background 0.3s",
      }}
    >
      <span style={{ fontSize: 7, color: "#6e7681", lineHeight: 1 }}>{label || idx}</span>
      <span
        style={{
          fontSize: 9,
          fontFamily: "monospace",
          color: changed
            ? direction === "up"
              ? "#7ee787"
              : "#ff7b72"
            : "#8b949e",
          fontWeight: changed ? 600 : 400,
        }}
      >
        {value.toFixed(2)}
      </span>
    </div>
  );
}

export function ObservationGroupInspector({ obs }: Props) {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const prevObsRef = useRef<Float32Array>(new Float32Array(215));

  useEffect(() => {
    if (obs) {
      // Delay prev update to show changes
      const timeout = setTimeout(() => {
        prevObsRef.current = new Float32Array(obs);
      }, 300);
      return () => clearTimeout(timeout);
    }
  }, [obs]);

  const toggleGroup = (name: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  if (!obs) {
    return (
      <div style={{ background: "#161b22", borderRadius: 6, border: "1px solid #21262d", padding: 10, color: "#8b949e", fontSize: 11 }}>
        Waiting for observation data...
      </div>
    );
  }

  const prev = prevObsRef.current;

  return (
    <div style={{ background: "#161b22", borderRadius: 6, border: "1px solid #21262d", padding: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#79c0ff", marginBottom: 6 }}>
        🔬 Observation Inspector <span style={{ fontWeight: 400, fontSize: 9, color: "#8b949e" }}>(215 features)</span>
      </div>
      {OBS_GROUPS.map((group) => {
        const expanded = expandedGroups.has(group.name);
        const size = group.end - group.start;

        // Count changed values in this group
        let changedCount = 0;
        for (let i = group.start; i < group.end; i++) {
          if (Math.abs(obs[i] - prev[i]) > 0.001) changedCount++;
        }

        // Get labels for entity slots
        let labels = group.labels;
        if (!labels) {
          if (group.name.startsWith("Hostile")) labels = HOSTILE_LABELS;
          else if (group.name.startsWith("Ally")) labels = ALLY_LABELS;
        }

        return (
          <div key={group.name} style={{ marginBottom: 2 }}>
            <div
              onClick={() => toggleGroup(group.name)}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "3px 6px",
                borderRadius: 4,
                cursor: "pointer",
                background: expanded ? "rgba(255,255,255,0.03)" : "transparent",
                borderLeft: `2px solid ${group.color}`,
              }}
            >
              <span style={{ fontSize: 10, color: group.color, fontWeight: 600 }}>
                {expanded ? "▾" : "▸"} {group.name}
                <span style={{ fontWeight: 400, color: "#6e7681", marginLeft: 4 }}>
                  [{group.start}..{group.end - 1}]
                </span>
              </span>
              <span style={{ fontSize: 9, color: changedCount > 0 ? "#ffa657" : "#484848" }}>
                {changedCount > 0 ? `${changedCount}Δ` : `${size}`}
              </span>
            </div>
            {expanded && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 1, padding: "4px 0 4px 8px" }}>
                {Array.from({ length: size }, (_, i) => {
                  const idx = group.start + i;
                  const label = labels?.[i];
                  return (
                    <ObsValue
                      key={idx}
                      value={obs[idx]}
                      prevValue={prev[idx]}
                      label={label}
                      idx={idx}
                    />
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}