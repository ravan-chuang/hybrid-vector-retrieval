import './FullCorpusEvaluation.css'

const PIPELINES = [
  { name: 'Sparse (C++ BM25)', ndcg: 0.222467, recall: 0.371466 },
  { name: 'Dense (HNSW)', ndcg: 0.345345, recall: 0.536151 },
  { name: 'Hybrid (Weighted RRF)', ndcg: 0.349124, recall: 0.545463 },
  { name: 'Dense + Cross-Encoder', ndcg: 0.428756, recall: 0.620690 },
  { name: 'Hybrid + Cross-Encoder', ndcg: 0.463473, recall: 0.674009 },
]

const ANN = [
  { method: 'Exact', p50: 12441.19, p95: 14343.13, recall: 1.0000, mrr: 0.3056, hit: 0.5781 },
  { method: 'HNSW ef=10', p50: 30.49, p95: 62.22, recall: 0.7596, mrr: 0.2252, hit: 0.4307 },
  { method: 'HNSW ef=20', p50: 41.87, p95: 83.50, recall: 0.8406, mrr: 0.2508, hit: 0.4778 },
  { method: 'HNSW ef=40', p50: 73.36, p95: 140.10, recall: 0.8924, mrr: 0.2673, hit: 0.5089 },
  { method: 'HNSW ef=80', p50: 105.58, p95: 193.52, recall: 0.9287, mrr: 0.2809, hit: 0.5311 },
  { method: 'HNSW ef=160', p50: 180.54, p95: 332.48, recall: 0.9517, mrr: 0.2890, hit: 0.5464 },
]

const ANN_BOOTSTRAP = [
  { ef: 10, delta: -0.080323, lower: -0.086050, upper: -0.074643 },
  { ef: 20, delta: -0.054763, lower: -0.059688, upper: -0.050013 },
  { ef: 40, delta: -0.038264, lower: -0.042486, upper: -0.034186 },
  { ef: 80, delta: -0.024649, lower: -0.028105, upper: -0.021338 },
  { ef: 160, delta: -0.016602, lower: -0.019482, upper: -0.013816 },
]


function LatencyMRRChart() {
  const width = 920
  const height = 390

  const margin = {
    top: 38,
    right: 48,
    bottom: 68,
    left: 76,
  }

  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom

  const minX = 20
  const maxX = 20000
  const minY = 0.20
  const maxY = 0.32

  const xScale = (value) =>
    margin.left +
    ((Math.log10(value) - Math.log10(minX)) /
      (Math.log10(maxX) - Math.log10(minX))) *
      plotWidth

  const yScale = (value) =>
    margin.top +
    ((maxY - value) / (maxY - minY)) *
      plotHeight

  const xTicks = [20, 50, 100, 200, 1000, 10000]
  const yTicks = [0.20, 0.23, 0.26, 0.29, 0.32]

  const hnsw = ANN.filter((item) => item.method !== 'Exact')
  const exact = ANN.find((item) => item.method === 'Exact')

  const linePath = hnsw
    .map((item, index) => {
      const x = xScale(item.p50)
      const y = yScale(item.mrr)
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`
    })
    .join(' ')

  return (
    <div className="fc-tradeoff-panel">
      <div className="fc-panel-heading">
        <h4>Latency–MRR Trade-off</h4>
        <span>
          6,980 dev queries · P50 SQL latency · log scale
        </span>
      </div>

      <div className="fc-tradeoff-legend">
        <span>
          <i className="fc-legend-dot fc-legend-hnsw" />
          HNSW
        </span>
        <span>
          <i className="fc-legend-dot fc-legend-exact" />
          Exact baseline
        </span>
      </div>

      <svg
        className="fc-tradeoff-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Latency versus MRR at 10 for five HNSW settings and Exact search"
      >
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={margin.left}
              x2={width - margin.right}
              y1={yScale(tick)}
              y2={yScale(tick)}
              className="fc-chart-grid"
            />
            <text
              x={margin.left - 14}
              y={yScale(tick) + 5}
              textAnchor="end"
              className="fc-chart-tick"
            >
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line
              x1={xScale(tick)}
              x2={xScale(tick)}
              y1={margin.top}
              y2={height - margin.bottom}
              className="fc-chart-grid"
            />
            <text
              x={xScale(tick)}
              y={height - margin.bottom + 25}
              textAnchor="middle"
              className="fc-chart-tick"
            >
              {tick >= 1000
                ? `${tick / 1000}k`
                : tick}
            </text>
          </g>
        ))}

        <path
          d={linePath}
          className="fc-chart-hnsw-line"
        />

        {hnsw.map((item, index) => {
          const x = xScale(item.p50)
          const y = yScale(item.mrr)

          return (
            <g key={item.method}>
              <circle
                cx={x}
                cy={y}
                r="7"
                className="fc-chart-hnsw-point"
              >
                <title>
                  {`${item.method}: P50 ${item.p50.toFixed(2)} ms, MRR@10 ${item.mrr.toFixed(4)}`}
                </title>
              </circle>

              <text
                x={x + 10}
                y={y - (index % 2 === 0 ? 13 : 16)}
                className="fc-chart-label"
              >
                {`ef=${item.method.split('=')[1]}`}
              </text>
            </g>
          )
        })}

        {exact && (
          <g>
            <circle
              cx={xScale(exact.p50)}
              cy={yScale(exact.mrr)}
              r="8"
              className="fc-chart-exact-point"
            >
              <title>
                {`Exact: P50 ${exact.p50.toFixed(2)} ms, MRR@10 ${exact.mrr.toFixed(4)}`}
              </title>
            </circle>

            <text
              x={xScale(exact.p50) - 12}
              y={yScale(exact.mrr) - 16}
              textAnchor="end"
              className="fc-chart-label"
            >
              Exact
            </text>
          </g>
        )}

        <text
          x={width / 2}
          y={height - 12}
          textAnchor="middle"
          className="fc-chart-axis-label"
        >
          P50 SQL Retrieval Latency (ms, logarithmic scale)
        </text>

        <text
          transform={`translate(22 ${height / 2}) rotate(-90)`}
          textAnchor="middle"
          className="fc-chart-axis-label"
        >
          MRR@10
        </text>
      </svg>

      <p className="fc-bootstrap-note">
        Frozen offline measurements on MS MARCO Dev Small.
        Query encoding and API overhead are excluded.
        Lines connect observed HNSW configurations;
        they do not represent a fitted performance model.
      </p>
    </div>
  )
}

function Metric({ label, value, detail }) {
  return (
    <div className="fc-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  )
}

function QualityBar({ item }) {
  return (
    <div className="fc-quality-row">
      <div className="fc-quality-heading">
        <span>{item.name}</span>
        <strong>{item.ndcg.toFixed(4)}</strong>
      </div>
      <div
        className="fc-bar-track"
        role="img"
        aria-label={`${item.name} nDCG@10 ${item.ndcg.toFixed(4)}`}
      >
        <div
          className="fc-bar-fill"
          style={{ width: `${item.ndcg * 100}%` }}
        />
      </div>
    </div>
  )
}

export default function FullCorpusEvaluation() {
  return (
    <section className="fc-section" aria-labelledby="fc-title">
      <div className="fc-header">
        <div>
          <span className="eyebrow">LARGE-SCALE EVALUATION</span>
          <h3 id="fc-title">Full MS MARCO · 8.84M Passages</h3>
          <p>
            Frozen offline experiments on the 8.84M corpus.
            These benchmark measurements are separate from
            live API request timings.
          </p>
        </div>
        <span className="fc-badge">OFFLINE · FROZEN</span>
      </div>

      <div className="fc-metrics">
        <Metric
          label="FULL CORPUS"
          value="8,841,823"
          detail="MS MARCO passages"
        />
        <Metric
          label="EMBEDDINGS"
          value="384D"
          detail="Normalized MiniLM"
        />
        <Metric
          label="HNSW INDEX"
          value="≈ 17 GB"
          detail="pgvector cosine ANN"
        />
        <Metric
          label="GIN INDEX"
          value="≈ 904 MB"
          detail="PostgreSQL full-text"
        />
      </div>

      <div className="fc-panels">
        <div className="fc-panel">
          <div className="fc-panel-heading">
            <h4>ANN Performance</h4>
            <span>6,980 dev queries · latency & relevance</span>
          </div>

          <div className="fc-ann-table-wrap">
            <table className="fc-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>P50</th>
                  <th>P95</th>
                  <th>ANN Recall@10</th>
                  <th>MRR@10</th>
                  <th>Hit@10</th>
                </tr>
              </thead>
              <tbody>
                {ANN.map((item) => (
                  <tr key={item.method}>
                    <td>{item.method}</td>
                    <td>{item.p50.toFixed(1)} ms</td>
                    <td>{item.p95.toFixed(1)} ms</td>
                    <td>{item.recall.toFixed(4)}</td>
                    <td>{item.mrr.toFixed(4)}</td>
                    <td>{item.hit.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="fc-highlight">
            <strong>≈ 169.6×</strong>
            <span>
              Lower P50 SQL latency at ef=40 vs Exact;
              ANN Recall@10 = 0.8924 · MRR@10 = 0.2673
            </span>
          </div>
        </div>

        <div className="fc-panel">
          <div className="fc-panel-heading">
            <h4>Retrieval Quality</h4>
            <span>6,980 dev queries · nDCG@10</span>
          </div>

          <div className="fc-quality-list">
            {PIPELINES.map((item) => (
              <QualityBar key={item.name} item={item} />
            ))}
          </div>

          <div className="fc-highlight">
            <strong>0.4635</strong>
            <span>
              Best nDCG@10 · sparse + dense candidates
              reranked by Cross-Encoder
            </span>
          </div>
        </div>
      </div>

      <div className="fc-bootstrap-panel">
        <div className="fc-panel-heading">
          <h4>ANN Relevance Loss · Paired Bootstrap</h4>
          <span>6,980 queries · 10,000 resamples · 95% CI</span>
        </div>

        <div className="fc-ann-table-wrap">
          <table className="fc-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>ΔMRR@10 vs Exact</th>
                <th>95% CI Lower</th>
                <th>95% CI Upper</th>
              </tr>
            </thead>
            <tbody>
              {ANN_BOOTSTRAP.map((item) => (
                <tr key={item.ef}>
                  <td>HNSW ef={item.ef}</td>
                  <td>{item.delta.toFixed(6)}</td>
                  <td>{item.lower.toFixed(6)}</td>
                  <td>{item.upper.toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="fc-bootstrap-note">
          Paired percentile bootstrap · seed 42.
          Negative ΔMRR indicates lower relevance than Exact.
          All five 95% confidence intervals exclude zero.
          SQL retrieval latency and relevance are measured separately.
        </p>
      </div>

      <LatencyMRRChart />

      <div className="fc-footer-metrics">
        <div>
          <span>HYBRID CE RECALL@10</span>
          <strong>0.6740</strong>
          <small>MS MARCO Dev Small</small>
        </div>
        <div>
          <span>HYBRID CE vs DENSE CE</span>
          <strong>+0.0347 nDCG@10</strong>
          <small>
            +8.1% · 95% paired bootstrap CI
            [+0.0306, +0.0390]
          </small>
        </div>
        <div>
          <span>RELEVANCE EVALUATION</span>
          <strong>6,980 queries</strong>
          <small>Official Dev Small qrels</small>
        </div>
      </div>

      <p className="fc-method-note">
        Retrieval quality is evaluated on MS MARCO Passage Dev Small
        (6,980 queries) using official relevance judgments with
        verified document-ID mapping. Results combine frozen
        historical experiment configurations: C++ BM25 v0.8.1,
        Dense HNSW, Weighted RRF, and offline Cross-Encoder reranking.
        Candidate-set sizes and implementation versions may differ;
        these values should not be interpreted as a controlled
        end-to-end serving benchmark. ANN Recall@10 measures
        agreement with exact dense neighbors, not human relevance.
        Cross-Encoder results are from offline evaluation,
        not live API reranking.
      </p>
    </section>
  )
}
