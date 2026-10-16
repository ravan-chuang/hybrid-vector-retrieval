import './ParallelBenchmarkCharts.css'

const WIDTH = 800
const HEIGHT = 280
const MARGIN = { top: 22, right: 26, bottom: 55, left: 68 }

const SERIES = [
  { key: 'baseline', label: 'v0.8.0', color: '#94a3b8' },
  { key: 'run1', label: 'v0.8.1 · Run 1', color: '#2563eb' },
  { key: 'run2', label: 'v0.8.1 · Run 2', color: '#0d9488' },
]

function Chart({ title, subtitle, rows, metric, speedup = false }) {
  const [hovered, setHovered] = React.useState(null)

  const series = speedup ? SERIES.slice(1) : SERIES

  const values = rows.flatMap((row) =>
    series.map((item) => row[item.key])
  )

  if (speedup) values.push(1)

  const maximum = Math.max(...values) * 1.14
  const minimum = speedup ? 0.8 : 0

  const plotWidth = WIDTH - MARGIN.left - MARGIN.right
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom

  const x = (index) =>
    MARGIN.left + (index / (rows.length - 1)) * plotWidth

  const y = (value) =>
    MARGIN.top +
    ((maximum - value) / (maximum - minimum)) * plotHeight

  const ticks = Array.from({ length: 5 }, (_, index) =>
    minimum + ((maximum - minimum) * index) / 4
  )

  return (
    <article className="pb-chart-card">
      <div className="pb-chart-heading">
        <h5>{title}</h5>
        <span>{subtitle}</span>
      </div>

      <div className="pb-chart-legend">
        {series.map((item) => (
          <span key={item.key}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>

      <svg
        className="pb-chart-svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={title}
      >
        {ticks.map((tick, index) => (
          <g key={index}>
            <line
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke="#e2e8f0"
              strokeDasharray="4 5"
            />
            <text
              x={MARGIN.left - 13}
              y={y(tick) + 4}
              textAnchor="end"
              fill="#64748b"
              fontSize="12"
            >
              {tick.toFixed(speedup ? 1 : 0)}
            </text>
          </g>
        ))}

        {speedup && (
          <g>
            <line
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={y(1)}
              y2={y(1)}
              stroke="#94a3b8"
              strokeDasharray="7 5"
            />
            <text
              x={WIDTH - MARGIN.right - 6}
              y={y(1) - 9}
              textAnchor="end"
              fill="#94a3b8"
              fontSize="11"
            >
              Baseline · 1.0×
            </text>
          </g>
        )}

        {rows.map((row, index) => (
          <text
            key={row.concurrency}
            x={x(index)}
            y={HEIGHT - 32}
            textAnchor="middle"
            fill="#64748b"
            fontSize="13"
          >
            {row.concurrency}
          </text>
        ))}

        <text
          x={MARGIN.left + plotWidth / 2}
          y={HEIGHT - 8}
          textAnchor="middle"
          fill="#64748b"
          fontSize="12"
          fontWeight="500"
        >
          Concurrent Clients
        </text>

        {series.map((item) => (
          <g key={item.key}>
            <polyline
              points={rows
                .map((row, index) =>
                  `${x(index)},${y(row[item.key])}`
                )
                .join(' ')}
              fill="none"
              stroke={item.color}
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {rows.map((row, index) => (
              <circle
                key={row.concurrency}
                cx={x(index)}
                cy={y(row[item.key])}
                r={hovered?.key === item.key &&
                   hovered?.concurrency === row.concurrency ? 7 : 4.5}
                fill={item.color}
                stroke="#ffffff"
                strokeWidth="2"
                tabIndex="0"
                onMouseEnter={() =>
                  setHovered({
                    key: item.key,
                    label: item.label,
                    concurrency: row.concurrency,
                    value: row[item.key],
                  })
                }
                onMouseLeave={() => setHovered(null)}
                onFocus={() =>
                  setHovered({
                    key: item.key,
                    label: item.label,
                    concurrency: row.concurrency,
                    value: row[item.key],
                  })
                }
                onBlur={() => setHovered(null)}
              >
                <title>
                  {`${item.label} · C${row.concurrency}: ${row[item.key].toFixed(2)}${speedup ? '×' : ''}`}
                </title>
              </circle>
            ))}
          </g>
        ))}
      </svg>

      <div className="pb-chart-tooltip" aria-live="polite">
        {hovered
          ? `${hovered.label} · Concurrency ${hovered.concurrency} · ${hovered.value.toFixed(2)}${speedup ? '×' : ''}`
          : 'Hover or focus a data point to inspect its value'}
      </div>
    </article>
  )
}

import React from 'react'

export default function ParallelBenchmarkCharts({ rows }) {
  const prepared = rows.map((row) => ({
    concurrency: row.concurrency,
    qps: {
      baseline: row.baselineQps,
      run1: row.run1Qps,
      run2: row.run2Qps,
    },
    p95: {
      baseline: row.baselineP95,
      run1: row.run1P95,
      run2: row.run2P95,
    },
    speedup: {
      baseline: 1,
      run1: row.run1Qps / row.baselineQps,
      run2: row.run2Qps / row.baselineQps,
    },
  }))

  return (
    <div className="pb-charts">
      <Chart
        title="HTTP Search Throughput"
        subtitle="Successful requests / second"
        rows={prepared.map((row) => ({
          concurrency: row.concurrency,
          ...row.qps,
        }))}
      />

      <Chart
        title="Client-Side P95 Latency"
        subtitle="Milliseconds · lower is better"
        rows={prepared.map((row) => ({
          concurrency: row.concurrency,
          ...row.p95,
        }))}
      />

      <Chart
        title="Throughput Speedup"
        subtitle="Relative to v0.8.0 baseline"
        rows={prepared.map((row) => ({
          concurrency: row.concurrency,
          ...row.speedup,
        }))}
        speedup
      />
    </div>
  )
}
