import React from "react";

const MOVE_LABELS = ["Hold", "Fwd", "FwdR", "Right", "BackR", "Back", "BackL", "Left", "FwdL"];
const COMBAT_LABELS = ["None", "Fire", "Reload", "Sw0", "Sw1", "Melee", "Block", "Dodge", "Repos"];
const TARGET_LABELS = ["T0", "T1", "T2", "T3", "Keep"];

interface Props {
  logits: { m: number[]; c: number[]; t: number[] } | null;
  masks: { m: boolean[]; c: boolean[]; t: boolean[] } | null;
  chosenAction: [number, number, number];
}

function softmax(logits: number[], mask: boolean[]): number[] {
  const masked = logits.map((v, i) => (mask[i] ? v : -1e8));
  const max = Math.max(...masked);
  const exps = masked.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => (sum > 0 ? v / sum : 0));
}

function ProbBar({ label, prob, chosen, masked }: { label: string; prob: number; chosen: boolean; masked: boolean }) {
  const pct = prob * 100;
  const color = masked
    ? "rgba(80,80,80,0.3)"
    : pct > 50
    ? "#7ee787"
    : pct > 20
    ? "#58a6ff"
    : pct > 5
    ? "#8b949e"
    : "rgba(139,148,158,0.3)";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, height: 16, fontSize: 9, fontFamily: "monospace" }}>
      <span style={{ width: 34, textAlign: "right", color: chosen ? "#fff" : "#8b949e", fontWeight: chosen ? 700 : 400 }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 10, background: "#161b22", borderRadius: 3, overflow: "hidden", position: "relative" }}>
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: 3,
            transition: "width 0.1s",
            boxShadow: chosen ? `0 0 6px ${color}` : "none",
          }}
        />
        {chosen && (
          <div
            style={{
              position: "absolute",
              right: 2,
              top: 0,
              height: "100%",
              display: "flex",
              alignItems: "center",
              color: "#fff",
              fontSize: 7,
              fontWeight: 700,
            }}
          >
            ◄
          </div>
        )}
      </div>
      <span style={{ width: 32, textAlign: "right", color: masked ? "#484848" : "#8b949e", fontSize: 8 }}>
        {masked ? "—" : `${pct.toFixed(0)}%`}
      </span>
    </div>
  );
}

function HeadSection({ title, labels, probs, mask, chosen, color }: {
  title: string; labels: string[]; probs: number[]; mask: boolean[]; chosen: number; color: string;
}) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color, marginBottom: 2, display: "flex", justifyContent: "space-between" }}>
        <span>{title}</span>
        <span style={{ color: "#8b949e", fontWeight: 400, fontSize: 9 }}>
          H={Math.max(...probs).toFixed(2)}
        </span>
      </div>
      {labels.map((label, i) => (
        <ProbBar key={i} label={label} prob={probs[i] || 0} chosen={i === chosen} masked={!mask[i]} />
      ))}
    </div>
  );
}

export function ActionProbabilityHeatmap({ logits, masks, chosenAction }: Props) {
  if (!logits || !masks) {
    return (
      <div style={{ background: "#161b22", borderRadius: 6, border: "1px solid #21262d", padding: 10, color: "#8b949e", fontSize: 11 }}>
        Load an ONNX model to see action probabilities.
      </div>
    );
  }

  const mProbs = softmax(logits.m, masks.m);
  const cProbs = softmax(logits.c, masks.c);
  const tProbs = softmax(logits.t, masks.t);

  // Entropy for each head (bits)
  const entropy = (probs: number[]) => -probs.reduce((s, p) => s + (p > 1e-8 ? p * Math.log2(p) : 0), 0);

  return (
    <div style={{ background: "#161b22", borderRadius: 6, border: "1px solid #21262d", padding: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#d2a8ff", marginBottom: 6, display: "flex", justifyContent: "space-between" }}>
        <span>🎯 Action Probabilities</span>
        <span style={{ fontWeight: 400, fontSize: 9, color: "#8b949e" }}>
          Ent: {(entropy(mProbs) + entropy(cProbs) + entropy(tProbs)).toFixed(2)} bits
        </span>
      </div>
      <HeadSection title="Movement" labels={MOVE_LABELS} probs={mProbs} mask={masks.m} chosen={chosenAction[0]} color="#7ee787" />
      <HeadSection title="Combat" labels={COMBAT_LABELS} probs={cProbs} mask={masks.c} chosen={chosenAction[1]} color="#ffa657" />
      <HeadSection title="Target" labels={TARGET_LABELS} probs={tProbs} mask={masks.t} chosen={chosenAction[2]} color="#d2a8ff" />
    </div>
  );
}
