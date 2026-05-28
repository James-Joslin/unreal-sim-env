import React, { useState } from "react";

export interface EpisodeResult {
  win: boolean;
  reward: number;
  kills: number;
  length: number;
}

export interface BatchResults {
  episodes: EpisodeResult[];
  winRate: number;
  meanReward: number;
  stdReward: number;
  meanKills: number;
  meanLength: number;
  rewardByOutcome: { wins: number; losses: number };
}

interface Props {
  onRunBatch: (n: number) => Promise<BatchResults>;
  disabled?: boolean;
}

function computeStats(episodes: EpisodeResult[]): BatchResults {
  const n = episodes.length;
  if (n === 0)
    return { episodes, winRate: 0, meanReward: 0, stdReward: 0, meanKills: 0, meanLength: 0, rewardByOutcome: { wins: 0, losses: 0 } };

  const wins = episodes.filter((e) => e.win);
  const losses = episodes.filter((e) => !e.win);

  const meanReward = episodes.reduce((s, e) => s + e.reward, 0) / n;
  const variance = episodes.reduce((s, e) => s + (e.reward - meanReward) ** 2, 0) / n;

  return {
    episodes,
    winRate: wins.length / n,
    meanReward,
    stdReward: Math.sqrt(variance),
    meanKills: episodes.reduce((s, e) => s + e.kills, 0) / n,
    meanLength: episodes.reduce((s, e) => s + e.length, 0) / n,
    rewardByOutcome: {
      wins: wins.length > 0 ? wins.reduce((s, e) => s + e.reward, 0) / wins.length : 0,
      losses: losses.length > 0 ? losses.reduce((s, e) => s + e.reward, 0) / losses.length : 0,
    },
  };
}

function StatBox({ label, value, color, sub }: { label: string; value: string; color: string; sub?: string }) {
  return (
    <div style={{ textAlign: "center", flex: 1 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color, fontFamily: "monospace" }}>{value}</div>
      <div style={{ fontSize: 9, color: "#8b949e" }}>{label}</div>
      {sub && <div style={{ fontSize: 8, color: "#6e7681" }}>{sub}</div>}
    </div>
  );
}

export function BatchEpisodeRunner({ onRunBatch, disabled }: Props) {
  const [results, setResults] = useState<BatchResults | null>(null);
  const [running, setRunning] = useState(false);
  const [batchSize, setBatchSize] = useState(50);

  const handleRun = async () => {
    setRunning(true);
    try {
      const raw = await onRunBatch(batchSize);
      setResults(computeStats(raw.episodes));
    } catch (err) {
      console.error("Batch run error:", err);
    }
    setRunning(false);
  };

  return (
    <div style={{ background: "#161b22", borderRadius: 6, border: "1px solid #21262d", padding: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#ffa657", marginBottom: 8 }}>
        🏃 Batch Episode Runner
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 8, alignItems: "center" }}>
        <button
          onClick={handleRun}
          disabled={running || disabled}
          style={{
            flex: 1,
            padding: "6px 12px",
            background: running ? "#21262d" : disabled ? "#21262d" : "#238636",
            color: running || disabled ? "#8b949e" : "#fff",
            border: "1px solid #30363d",
            borderRadius: 6,
            cursor: running || disabled ? "default" : "pointer",
            fontSize: 11,
            fontFamily: "inherit",
            fontWeight: 600,
          }}
        >
          {running ? "⏳ Running..." : `▶ Run ${batchSize} Episodes`}
        </button>
        <select
          value={batchSize}
          onChange={(e) => setBatchSize(Number(e.target.value))}
          disabled={running}
          style={{
            padding: "4px 6px",
            background: "#0d1117",
            color: "#c9d1d9",
            border: "1px solid #30363d",
            borderRadius: 4,
            fontSize: 11,
            fontFamily: "inherit",
          }}
        >
          <option value={10}>10</option>
          <option value={25}>25</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </div>

      {disabled && (
        <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 6 }}>
          Load an ONNX model to enable batch testing.
        </div>
      )}

      {results && (
        <>
          {/* Big stat boxes */}
          <div style={{ display: "flex", gap: 4, marginBottom: 8, padding: "8px 0", borderTop: "1px solid #21262d", borderBottom: "1px solid #21262d" }}>
            <StatBox
              label="Win Rate"
              value={`${(results.winRate * 100).toFixed(0)}%`}
              color={results.winRate > 0.5 ? "#7ee787" : results.winRate > 0.3 ? "#ffa657" : "#f85149"}
            />
            <StatBox
              label="Avg Reward"
              value={results.meanReward.toFixed(0)}
              color="#58a6ff"
              sub={`±${results.stdReward.toFixed(0)}`}
            />
            <StatBox
              label="Avg Kills"
              value={results.meanKills.toFixed(1)}
              color="#d2a8ff"
            />
            <StatBox
              label="Avg Length"
              value={results.meanLength.toFixed(0)}
              color="#8b949e"
            />
          </div>

          {/* Reward by outcome */}
          <div style={{ fontSize: 10, color: "#8b949e", lineHeight: 1.8 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Win reward avg:</span>
              <span style={{ color: "#7ee787", fontFamily: "monospace" }}>
                {results.rewardByOutcome.wins > 0 ? `+${results.rewardByOutcome.wins.toFixed(0)}` : "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Loss reward avg:</span>
              <span style={{ color: "#ffa657", fontFamily: "monospace" }}>
                {results.rewardByOutcome.losses !== 0 ? results.rewardByOutcome.losses.toFixed(0) : "—"}
              </span>
            </div>
            {results.rewardByOutcome.losses > results.rewardByOutcome.wins && results.rewardByOutcome.wins > 0 && (
              <div style={{ color: "#f85149", fontSize: 9, marginTop: 4, padding: "3px 6px", background: "rgba(248,81,73,0.1)", borderRadius: 3 }}>
                ⚠️ Losses more rewarding than wins — reward signal is misaligned
              </div>
            )}
          </div>

          {/* Mini histogram of reward distribution */}
          <div style={{ marginTop: 8, fontSize: 9, color: "#6e7681" }}>
            <div style={{ marginBottom: 2 }}>Reward Distribution:</div>
            <div style={{ display: "flex", height: 30, gap: 1, alignItems: "flex-end" }}>
              {(() => {
                const rewards = results.episodes.map((e) => e.reward);
                const min = Math.min(...rewards);
                const max = Math.max(...rewards);
                const buckets = 20;
                const step = (max - min) / buckets || 1;
                const counts = new Array(buckets).fill(0);
                for (const r of rewards) {
                  const idx = Math.min(Math.floor((r - min) / step), buckets - 1);
                  counts[idx]++;
                }
                const maxCount = Math.max(...counts, 1);

                return counts.map((count, i) => {
                  const bucketMid = min + (i + 0.5) * step;
                  const isWinBucket = bucketMid > results.rewardByOutcome.wins * 0.8;
                  return (
                    <div
                      key={i}
                      style={{
                        flex: 1,
                        height: `${(count / maxCount) * 100}%`,
                        minHeight: count > 0 ? 2 : 0,
                        background: isWinBucket ? "#7ee787" : "#58a6ff",
                        borderRadius: "2px 2px 0 0",
                        opacity: 0.7,
                      }}
                      title={`${(min + i * step).toFixed(0)} - ${(min + (i + 1) * step).toFixed(0)}: ${count} episodes`}
                    />
                  );
                });
              })()}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
