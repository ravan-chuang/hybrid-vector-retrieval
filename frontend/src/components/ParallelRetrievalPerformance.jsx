import benchmark from '../data/parallelRetrievalBenchmark.json'
import ParallelBenchmarkCharts from './ParallelBenchmarkCharts'
import './FullCorpusEvaluation.css'

const baseline = benchmark.experiments.v080_gil_release.summary
const parallelRun1 = benchmark.experiments.v081_parallel_run1.summary
const parallelRun2 = benchmark.experiments.v081_parallel_run2.summary

function formatNumber(value, digits = 2) {
  return Number(value).toFixed(digits)
}

export default function ParallelRetrievalPerformance() {
  const rows = baseline.map((base, index) => {
    const run1 = parallelRun1[index]
    const run2 = parallelRun2[index]

    if (
      base.concurrency !== run1.concurrency ||
      base.concurrency !== run2.concurrency
    ) {
      throw new Error('Benchmark concurrency mismatch')
    }

    return {
      concurrency: base.concurrency,
      baselineQps: base.successful_qps,
      run1Qps: run1.successful_qps,
      run2Qps: run2.successful_qps,
      baselineP95: base.client_ms.p95,
      run1P95: run1.client_ms.p95,
      run2P95: run2.client_ms.p95,
      speedup: run1.successful_qps / base.successful_qps,
    }
  })

  const c16 = rows.find((row) => row.concurrency === 16)

  const p95Reduction =
    (1 - c16.run1P95 / c16.baselineP95) * 100

  return (
    <section className="fc-section" aria-labelledby="parallel-title">
      <div className="fc-header">
        <div>
          <span className="eyebrow">
            PARALLEL RETRIEVAL PERFORMANCE
          </span>
          <h3 id="parallel-title">
            C++ BM25 · v0.8.0 vs v0.8.1
          </h3>
          <p>
            Frozen local HTTP benchmarks on 8.84 million
            MS MARCO passages. These results measure the
            complete Hybrid Search API, not BM25 alone.
          </p>
        </div>

        <span className="fc-badge">
          HTTP · FROZEN
        </span>
      </div>

      <div className="fc-metrics">
        <div className="fc-metric">
          <span>PEAK MEASURED THROUGHPUT</span>
          <strong>
            {formatNumber(
              Math.max(...parallelRun1.map(
                (row) => row.successful_qps
              ))
            )} QPS
          </strong>
          <small>v0.8.1 · Run 1</small>
        </div>

        <div className="fc-metric">
          <span>THROUGHPUT SPEEDUP</span>
          <strong>{formatNumber(c16.speedup)}×</strong>
          <small>Concurrency 16 · Run 1 vs baseline</small>
        </div>

        <div className="fc-metric">
          <span>P95 LATENCY REDUCTION</span>
          <strong>{formatNumber(p95Reduction, 1)}%</strong>
          <small>Concurrency 16 · Run 1 vs baseline</small>
        </div>

        <div className="fc-metric">
          <span>SUCCESSFUL REQUESTS</span>
          <strong>7,500</strong>
          <small>3 runs · 0 failures</small>
        </div>
      </div>

      <ParallelBenchmarkCharts rows={rows} />

      <div className="fc-panel">
        <div className="fc-panel-heading">
          <h4>Throughput &amp; Tail Latency</h4>
          <span>500 requests / concurrency / run</span>
        </div>

        <div className="fc-ann-table-wrap">
          <table className="fc-table parallel-performance-table">
            <colgroup>
              <col style={{ width: '13%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '15%' }} />
            </colgroup>
            <thead>
              <tr>
                <th rowSpan="2">Concurrency</th>
                <th colSpan="3">Throughput (QPS)</th>
                <th colSpan="3">Client P95 (ms)</th>
                <th rowSpan="2">Speedup</th>
              </tr>
              <tr>
                <th>v0.8.0</th>
                <th>v0.8.1 R1</th>
                <th>v0.8.1 R2</th>
                <th>v0.8.0</th>
                <th>v0.8.1 R1</th>
                <th>v0.8.1 R2</th>
              </tr>
            </thead>

            <tbody>
              {rows.map((row) => (
                <tr key={row.concurrency}>
                  <td>{row.concurrency}</td>
                  <td>{formatNumber(row.baselineQps)}</td>
                  <td>{formatNumber(row.run1Qps)}</td>
                  <td>{formatNumber(row.run2Qps)}</td>
                  <td>{formatNumber(row.baselineP95)}</td>
                  <td>{formatNumber(row.run1P95)}</td>
                  <td>{formatNumber(row.run2P95)}</td>
                  <td>{formatNumber(row.speedup)}×</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="fc-method-note">
        Workload: /search/hybrid · candidate_k=50 ·
        ef_search=160 · top_k=10 · 20 warmup requests
        per concurrency level. v0.8.0 releases the Python
        GIL but serializes BM25 searches; v0.8.1 uses four
        independent BM25 search workspaces. Each run includes
        2,500 measured requests. Results are local-machine
        observations, not production throughput guarantees.
      </p>
    </section>
  )
}
