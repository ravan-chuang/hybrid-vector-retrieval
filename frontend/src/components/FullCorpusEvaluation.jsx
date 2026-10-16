import './FullCorpusEvaluation.css'

const PIPELINES = [
  { name: 'Lexical', ndcg: 0.1940, recall: 0.2982 },
  { name: 'Dense HNSW', ndcg: 0.3741, recall: 0.5879 },
  { name: 'Weighted RRF', ndcg: 0.3863, recall: 0.6010 },
  { name: 'Dense + Cross-Encoder', ndcg: 0.4703, recall: 0.6950 },
  { name: 'Union + Cross-Encoder', ndcg: 0.4876, recall: 0.7211 },
]

const ANN = [
  { method: 'Exact', p50: 5751.5, recall: 1.0 },
  { method: 'HNSW ef=40', p50: 31.9, recall: 0.858 },
  { method: 'HNSW ef=160', p50: 95.6, recall: 0.939 },
]

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
            <span>100 queries · P50 latency</span>
          </div>

          <div className="fc-ann-table-wrap">
            <table className="fc-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>P50</th>
                  <th>ANN Recall@10</th>
                </tr>
              </thead>
              <tbody>
                {ANN.map((item) => (
                  <tr key={item.method}>
                    <td>{item.method}</td>
                    <td>{item.p50.toFixed(1)} ms</td>
                    <td>{item.recall.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="fc-highlight">
            <strong>≈ 180×</strong>
            <span>
              Lower P50 latency at ef=40 vs exact;
              ANN Recall@10 = 0.858
            </span>
          </div>
        </div>

        <div className="fc-panel">
          <div className="fc-panel-heading">
            <h4>Retrieval Quality</h4>
            <span>10,000 queries · nDCG@10</span>
          </div>

          <div className="fc-quality-list">
            {PIPELINES.map((item) => (
              <QualityBar key={item.name} item={item} />
            ))}
          </div>

          <div className="fc-highlight">
            <strong>0.4876</strong>
            <span>
              Best nDCG@10 · lexical + dense candidates
              reranked by Cross-Encoder
            </span>
          </div>
        </div>
      </div>

      <div className="fc-footer-metrics">
        <div>
          <span>UNION CE RECALL@10</span>
          <strong>0.7211</strong>
        </div>
        <div>
          <span>UNION CE vs DENSE CE</span>
          <strong>+0.0173 nDCG@10</strong>
          <small>95% CI [+0.0147, +0.0199]</small>
        </div>
        <div>
          <span>LEXICAL RESCUE</span>
          <strong>339 / 485</strong>
          <small>69.9% promoted into final Top-10</small>
        </div>
      </div>

      <p className="fc-method-note">
        Relevance metrics use binary positive judgments from the
        MS MARCO labeled-list training split, not official dev
        leaderboard qrels. ANN Recall@10 measures agreement with
        exact dense neighbors, not human relevance.
        These are historical offline evaluation results.
        The lexical backend and candidate-generation settings
        have not yet been verified against the current
        C++ BM25 v0.8.1 serving pipeline.
        Cross-Encoder results are from offline evaluation,
        not live API reranking.
      </p>
    </section>
  )
}
