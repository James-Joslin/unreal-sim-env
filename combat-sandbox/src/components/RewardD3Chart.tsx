import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";

export interface RewardStepLog {
  step: number;
  action: [number, number, number];
  actionDesc: string;
  reward: number;
  breakdown: {
    damageDealt: number;
    killBonus: number;
    damageTaken: number;
    timePenalty: number;
    optimalRange: number;
    rangeClosing: number;
    outOfRange: number;
    flanking: number;
    inactivity: number;
    weaponSelection: number;
    ammo: number;
    endBonus: number;
    engagement: number;
  };
  cumReward: number;
}

interface RewardD3ChartProps {
  history: RewardStepLog[];
  stage: number;
}

export function RewardD3Chart({ history, stage }: RewardD3ChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [chartMode, setChartMode] = useState<"instant" | "cumulative">("cumulative");
  const [hoveredPoint, setHoveredPoint] = useState<RewardStepLog | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [width, setWidth] = useState(480);

  // Resize listener to ensure responsive design
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 50) {
          setWidth(entry.contentRect.width);
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const height = 180;
  const margin = { top: 15, right: 15, bottom: 25, left: 40 };

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    if (history.length === 0) {
      return;
    }

    const valueFn = (d: RewardStepLog) =>
      chartMode === "cumulative" ? d.cumReward : d.reward;

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain(d3.extent(history, (d) => d.step) as [number, number])
      .range([margin.left, width - margin.right]);

    const yExtent = d3.extent(history, valueFn) as [number, number];
    // Give some vertical padding
    const yPadding = Math.abs(yExtent[1] - yExtent[0]) * 0.15 || 1.0;
    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([height - margin.bottom, margin.top]);

    // Gridlines
    const yTicks = 4;
    svg
      .append("g")
      .attr("class", "grid")
      .attr("transform", `translate(${margin.left},0)`)
      .call(
        d3
          .axisLeft(yScale)
          .ticks(yTicks)
          .tickSize(-width + margin.left + margin.right)
          .tickFormat(() => "")
      )
      .call((g) => g.select(".domain").remove())
      .call((g) =>
        g
          .selectAll(".tick line")
          .attr("stroke", "#21262d")
          .attr("stroke-dasharray", "2,2")
      );

    // X-Axis
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(Math.min(5, history.length))
      .tickFormat((d) => `S${d}`);

    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(xAxis)
      .call((g) => {
        g.select(".domain").attr("stroke", "#30363d");
        g.selectAll(".tick line").attr("stroke", "#30363d");
        g.selectAll(".tick text")
          .attr("fill", "#8b949e")
          .style("font-family", "monospace")
          .style("font-size", "10px");
      });

    // Y-Axis
    const yAxis = d3.axisLeft(yScale).ticks(yTicks).tickFormat(d3.format(".1f"));
    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(yAxis)
      .call((g) => {
        g.select(".domain").attr("stroke", "#30363d");
        g.selectAll(".tick line").attr("stroke", "#30363d");
        g.selectAll(".tick text")
          .attr("fill", "#8b949e")
          .style("font-family", "monospace")
          .style("font-size", "10px");
      });

    // Gradient definitions
    const gradientId = "reward-gradient-" + chartMode;
    const gradient = svg
      .append("defs")
      .append("linearGradient")
      .attr("id", gradientId)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "0%")
      .attr("y2", "100%");

    const colorColor = chartMode === "cumulative" ? "#58a6ff" : "#7ee787";

    gradient
      .append("stop")
      .attr("offset", "0%")
      .attr("stop-color", colorColor)
      .attr("stop-opacity", 0.25);
    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colorColor)
      .attr("stop-opacity", 0.0);

    // Area
    const area = d3
      .area<RewardStepLog>()
      .x((d) => xScale(d.step))
      .y0(height - margin.bottom)
      .y1((d) => yScale(valueFn(d)))
      .curve(d3.curveMonotoneX);

    svg
      .append("path")
      .datum(history)
      .attr("fill", `url(#${gradientId})`)
      .attr("d", area);

    // Line Path
    const line = d3
      .line<RewardStepLog>()
      .x((d) => xScale(d.step))
      .y((d) => yScale(valueFn(d)))
      .curve(d3.curveMonotoneX);

    svg
      .append("path")
      .datum(history)
      .attr("fill", "none")
      .attr("stroke", colorColor)
      .attr("stroke-width", 2)
      .attr("d", line);

    // Hover guidance elements Group
    const hoverGroup = svg.append("g").style("display", "none");

    const verticalLine = hoverGroup
      .append("line")
      .attr("stroke", "#8b949e")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3,3")
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom);

    const activePoint = hoverGroup
      .append("circle")
      .attr("r", 5)
      .attr("fill", colorColor)
      .attr("stroke", "#0d1117")
      .attr("stroke-width", 1.5);

    // Interaction overlay
    svg
      .append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "transparent")
      .style("cursor", "crosshair")
      .on("pointerenter pointermove", (event) => {
        const [mX] = d3.pointer(event);
        const xVal = xScale.invert(mX);

        // Find closest point in step
        const bisect = d3.bisector((d: RewardStepLog) => d.step).left;
        const index = bisect(history, xVal, 1);
        const d0 = history[index - 1];
        const d1 = history[index];
        let d = d0;
        if (d1 && xVal - d0.step > d1.step - xVal) {
          d = d1;
        }

        if (d) {
          setHoveredPoint(d);
          const px = xScale(d.step);
          const py = yScale(valueFn(d));

          verticalLine.attr("x1", px).attr("x2", px);
          activePoint.attr("cx", px).attr("cy", py);
          hoverGroup.style("display", null);

          // Position tooltip relative to container
          const rect = svgEl.getBoundingClientRect();
          setTooltipPos({
            x: px + 15,
            y: py - 10,
          });
        }
      })
      .on("pointerleave", () => {
        hoverGroup.style("display", "none");
        setHoveredPoint(null);
        setTooltipPos(null);
      });
  }, [history, chartMode, width]);

  return (
    <div
      ref={containerRef}
      style={{
        background: "#161b22",
        borderRadius: 8,
        border: "1px solid #21262d",
        padding: "10px 14px",
        marginTop: 12,
        position: "relative",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 700, color: "#58a6ff" }}>
          📈 Reward Analysis (Stage {stage})
        </span>
        <div style={{ display: "flex", gap: 4 }}>
          <button
            onClick={() => setChartMode("cumulative")}
            style={{
              ...tabBtn,
              background: chartMode === "cumulative" ? "#21262d" : "transparent",
              color: chartMode === "cumulative" ? "#58a6ff" : "#8b949e",
              border: chartMode === "cumulative" ? "1px solid #30363d" : "1px solid transparent",
            }}
          >
            Cumulative
          </button>
          <button
            onClick={() => setChartMode("instant")}
            style={{
              ...tabBtn,
              background: chartMode === "instant" ? "#21262d" : "transparent",
              color: chartMode === "instant" ? "#7ee787" : "#8b949e",
              border: chartMode === "instant" ? "1px solid #30363d" : "1px solid transparent",
            }}
          >
            Instant
          </button>
        </div>
      </div>

      {history.length === 0 ? (
        <div
          style={{
            height: height,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#8b949e",
            fontSize: 11,
          }}
        >
          No combat reward logs yet. Tick the simulation to view step-by-step telemetry.
        </div>
      ) : (
        <svg
          ref={svgRef}
          style={{
            width: "100%",
            height: height,
            overflow: "visible",
            display: "block",
          }}
        />
      )}

      {/* Tooltip Overlay */}
      {hoveredPoint && tooltipPos && (
        <div
          style={{
            position: "absolute",
            left: Math.min(tooltipPos.x, width - 190),
            top: Math.max(10, tooltipPos.y - 70),
            width: 170,
            background: "#0d1117",
            border: "1px solid #30363d",
            borderRadius: 6,
            padding: "8px 10px",
            fontSize: 10,
            fontFamily: "monospace",
            pointerEvents: "none",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            zIndex: 10,
          }}
        >
          <div style={{ borderBottom: "1px solid #21262d", paddingBottom: 4, marginBottom: 4 }}>
            <div style={{ color: "#8b949e" }}>Step {hoveredPoint.step}</div>
            <div style={{ color: "#c9d1d9", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
              {hoveredPoint.actionDesc}
            </div>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
            <span style={{ color: "#8b949e" }}>Reward:</span>
            <span style={{ color: hoveredPoint.reward >= 0 ? "#7ee787" : "#ff7b72", fontWeight: "bold" }}>
              {hoveredPoint.reward >= 0 ? "+" : ""}
              {hoveredPoint.reward.toFixed(2)}
            </span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ color: "#8b949e" }}>Cumulative:</span>
            <span style={{ color: "#58a6ff" }}>{hoveredPoint.cumReward.toFixed(2)}</span>
          </div>

          <div style={{ fontStyle: "italic", color: "#8b949e", fontSize: 9, marginBottom: 2, borderTop: "1px solid #21262d", paddingTop: 3 }}>
            Contributions:
          </div>
          {Object.entries(hoveredPoint.breakdown)
            .filter(([_, val]) => typeof val === "number" && val !== 0)
            .map(([key, v]) => {
              const val = v as number;
              const label = key
                .replace(/([A-Z])/g, " $1")
                .replace(/^./, (str) => str.toUpperCase());
              return (
                <div key={key} style={{ display: "flex", justifyContent: "space-between", fontSize: 9, padding: "1px 0" }}>
                  <span style={{ color: "#8b949e" }}>{label}:</span>
                  <span style={{ color: val > 0 ? "#7ee787" : "#ff7b72" }}>
                    {val > 0 ? "+" : ""}
                    {val.toFixed(2)}
                  </span>
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}

const tabBtn = {
  padding: "2px 8px",
  borderRadius: 4,
  fontSize: 10,
  cursor: "pointer",
  transition: "all 0.1s",
  fontFamily: "inherit",
};
