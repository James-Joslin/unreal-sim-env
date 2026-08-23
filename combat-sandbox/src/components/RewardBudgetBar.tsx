import React from "react";

interface Props {
  cumReward: number;
  step: number;
  maxSteps: number;
  numTargets: number;
  kills: number;
  agentAlive: boolean;
  done: boolean;
  stage: number;
}

// Compute theoretical reward budgets (matches Python reward_budget_analysis)
function computeBudgets(numTargets: number, maxSteps: number, stage: number) {
  const RW = {
    damage_dealt: 0.15,            // Synced with reward.py (was 0.25)
    kill_target: 35.0,             // Synced with reward.py (was 20.0)
    target_low_hp: 3.0,
    episode_win: 50.0,
    episode_timeout: -8.0,
    surviving_target: -8.0,
    die: -10.0,
    episode_loss: -5.0,
    take_damage: -0.015,
    alive_per_step: -0.02,
  };

  const winSteps = Math.min(Math.round(maxSteps * 0.3), 200);
  const win =
    RW.damage_dealt * 100 * numTargets +
    RW.kill_target * numTargets +
    RW.target_low_hp * numTargets +
    RW.episode_win * 1.3 +
    RW.take_damage * 60 +
    RW.alive_per_step * winSteps +
    8; // shaping estimate

  const timeout =
    RW.damage_dealt * 100 * 0.4 +
    RW.target_low_hp +
    RW.episode_timeout * Math.max(1.0, maxSteps / 200.0) +
    RW.surviving_target * numTargets +
    RW.take_damage * 50 +
    RW.alive_per_step * maxSteps +
    5;

  const deathSteps = 80;
  const death =
    RW.damage_dealt * 100 * 1.5 +
    RW.kill_target * 1 +
    RW.target_low_hp +
    RW.die +
    RW.episode_loss +
    RW.surviving_target * (numTargets - 1) +
    RW.take_damage * 100 +
    RW.alive_per_step * deathSteps +
    3;

  return { win, timeout, death };
}

function BudgetRow({ label, value, color, isCurrent }: { label: string; value: number; color: string; isCurrent?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "2px 0", fontSize: 10 }}>
      <span style={{ color: isCurrent ? "#fff" : "#8b949e", fontWeight: isCurrent ? 700 : 400 }}>
        {isCurrent ? "▶ " : "  "}{label}
      </span>
      <span style={{ color, fontWeight: 600, fontFamily: "monospace" }}>
        {value >= 0 ? "+" : ""}{value.toFixed(0)}
      </span>
    </div>
  );
}

export function RewardBudgetBar({ cumReward, step, maxSteps, numTargets, kills, agentAlive, done, stage }: Props) {
  const budgets = computeBudgets(numTargets, maxSteps, stage);

  // Determine which scenario the episode is tracking toward
  let trajectory = "fighting";
  let trajectoryColor = "#8b949e";
  if (done) {
    if (!agentAlive) { trajectory = "DEATH"; trajectoryColor = "#f85149"; }
    else if (kills >= numTargets) { trajectory = "WIN"; trajectoryColor = "#7ee787"; }
    else { trajectory = "TIMEOUT"; trajectoryColor = "#ffa657"; }
  } else {
    // Live prediction based on current reward vs budgets
    const progress = step / maxSteps;
    const projectedReward = progress > 0 ? cumReward / progress : cumReward;
    if (projectedReward > budgets.win * 0.7) { trajectory = "→ Win track"; trajectoryColor = "#7ee787"; }
    else if (projectedReward > budgets.death * 0.8) { trajectory = "→ Fighting"; trajectoryColor = "#58a6ff"; }
    else if (projectedReward < budgets.timeout * 0.5) { trajectory = "→ Struggling"; trajectoryColor = "#ffa657"; }
    else { trajectory = "→ Uncertain"; trajectoryColor = "#8b949e"; }
  }

  // Warning if reward exceeds win budget (potential farming)
  const isFarming = cumReward > budgets.win * 1.5 && kills < numTargets;

  // Visual gauge: show cumReward position relative to budget range
  const minBudget = Math.min(budgets.timeout, budgets.death, -50);
  const maxBudget = Math.max(budgets.win, cumReward, 50);
  const range = maxBudget - minBudget;
  const cumPos = ((cumReward - minBudget) / range) * 100;
  const winPos = ((budgets.win - minBudget) / range) * 100;
  const deathPos = ((budgets.death - minBudget) / range) * 100;
  const timeoutPos = ((budgets.timeout - minBudget) / range) * 100;

  return (
    <div style={{ background: "#161b22", borderRadius: 6, border: "1px solid #21262d", padding: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#58a6ff" }}>📊 Reward Budget</span>
        <span style={{ fontSize: 10, color: trajectoryColor, fontWeight: 600 }}>{trajectory}</span>
      </div>

      {/* Visual gauge */}
      <div style={{ position: "relative", height: 20, background: "#0d1117", borderRadius: 4, marginBottom: 8, overflow: "hidden" }}>
        {/* Budget markers */}
        <div style={{ position: "absolute", left: `${timeoutPos}%`, top: 0, height: "100%", width: 2, background: "#f85149", opacity: 0.6 }} />
        <div style={{ position: "absolute", left: `${deathPos}%`, top: 0, height: "100%", width: 2, background: "#ffa657", opacity: 0.6 }} />
        <div style={{ position: "absolute", left: `${winPos}%`, top: 0, height: "100%", width: 2, background: "#7ee787", opacity: 0.6 }} />

        {/* Current reward position */}
        <div
          style={{
            position: "absolute",
            left: `${Math.max(0, Math.min(cumPos, 100))}%`,
            top: 2,
            width: 8,
            height: 16,
            background: isFarming ? "#f85149" : "#58a6ff",
            borderRadius: 2,
            transform: "translateX(-4px)",
            boxShadow: `0 0 6px ${isFarming ? "#f85149" : "#58a6ff"}`,
            transition: "left 0.15s",
          }}
        />
      </div>

      {/* Labels under gauge */}
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#6e7681", marginBottom: 6 }}>
        <span>TIMEOUT</span>
        <span>DEATH</span>
        <span>WIN</span>
      </div>

      {/* Budget values */}
      <BudgetRow label="Win (all kills)" value={budgets.win} color="#7ee787" />
      <BudgetRow label="Death (1 kill)" value={budgets.death} color="#ffa657" />
      <BudgetRow label="Timeout (0 kills)" value={budgets.timeout} color="#f85149" />
      <div style={{ borderTop: "1px solid #21262d", marginTop: 4, paddingTop: 4 }}>
        <BudgetRow label={`Current (${kills} kills)`} value={cumReward} color={isFarming ? "#f85149" : "#58a6ff"} isCurrent />
      </div>

      {isFarming && (
        <div style={{
          marginTop: 6, padding: "4px 8px", background: "rgba(248,81,73,0.1)", border: "1px solid rgba(248,81,73,0.3)",
          borderRadius: 4, fontSize: 9, color: "#ff7b72",
        }}>
          ⚠️ Reward exceeds WIN budget by {((cumReward / budgets.win - 1) * 100).toFixed(0)}% with {kills}/{numTargets} kills — possible farming
        </div>
      )}

      <div style={{ fontSize: 9, color: "#6e7681", marginTop: 4 }}>
        Step {step}/{maxSteps} • Stage {stage} • {numTargets} targets
      </div>
    </div>
  );
}