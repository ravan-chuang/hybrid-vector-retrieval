import { useEffect, useState } from 'react'
import { api } from './api'
import DocumentManager from './components/DocumentManager'
import './App.css'

const METHODS = [
  { id: 'hybrid', label: 'Hybrid' },
  { id: 'vector', label: 'Vector' },
  { id: 'lexical', label: 'Lexical' },
]

function formatNumber(value) {
  if (value === null || value === undefined) return '—'
  return Number(value).toLocaleString()
}

function formatMs(value) {
  if (value === null || value === undefined) return '—'
  return `${Number(value).toFixed(2)} ms`
}

function StatCard({ label, value, detail }) {
  return (
    <article className="stat-card">
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
      <span className="stat-detail">{detail}</span>
    </article>
  )
}

function App() {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState([])
  const [method, setMethod] = useState('hybrid')
  const [query, setQuery] = useState('Manhattan Project')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [systemLoading, setSystemLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    Promise.all([
      api.health(),
      api.stats(),
      api.history(),
    ])
      .then(([healthData, statsData, historyData]) => {
        if (cancelled) return
        setHealth(healthData)
        setStats(statsData)
        setHistory(historyData)
      })
      .catch((err) => {
        if (!cancelled) {
          setError(`Backend connection failed: ${err.message}`)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setSystemLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  async function handleSearch(event) {
    event.preventDefault()
    const cleanQuery = query.trim()
    if (!cleanQuery) return

    setLoading(true)
    setError('')

    try {
      let data

      if (method === 'vector') {
        data = await api.vectorSearch(cleanQuery)
      } else if (method === 'lexical') {
        data = await api.lexicalSearch(cleanQuery)
      } else {
        data = await api.hybridSearch(cleanQuery)
      }

      setResult(data)

      try {
        setHistory(await api.history())
      } catch {
        // Search results remain usable even if history refresh fails.
      }
    } catch (err) {
      setError(`Search failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const indexes = stats?.indexes ?? []
  const hnsw = indexes.find((item) =>
    item.name.toLowerCase().includes('hnsw'),
  )
  const gin = indexes.find((item) =>
    item.name.toLowerCase().includes('gin'),
  )

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">HV</div>
          <div>
            <h1>Hybrid Vector Retrieval</h1>
            <p>PostgreSQL · pgvector · Information Retrieval</p>
          </div>
        </div>

        <div
          className={`status-pill ${
            health?.database_check ? 'online' : 'offline'
          }`}
        >
          <span className="status-dot" />
          {health?.database_check ? 'Database online' : 'Database offline'}
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <span className="eyebrow">RETRIEVAL ENGINEERING DASHBOARD</span>
            <h2>
              Lexical and semantic search,
              <br />
              unified in PostgreSQL.
            </h2>
            <p>
              Explore a 500K-document retrieval system backed by PostgreSQL
              full-text search, pgvector HNSW, and Reciprocal Rank Fusion.
            </p>
          </div>

          <div className="hero-meta">
            <span>Embedding model</span>
            <strong>
              {stats?.embedding_model ??
                'sentence-transformers/all-MiniLM-L6-v2'}
            </strong>
            <span>{stats?.embedding_dimension ?? 384} dimensions</span>
          </div>
        </section>

        {error && <div className="error-banner">{error}</div>}

        <section className="stats-grid">
          <StatCard
            label="DOCUMENTS"
            value={
              systemLoading ? '…' : formatNumber(stats?.documents)
            }
            detail="MS MARCO passages"
          />
          <StatCard
            label="DATABASE"
            value={stats?.database ?? 'PostgreSQL'}
            detail={stats?.vector_extension ?? 'pgvector'}
          />
          <StatCard
            label="HNSW INDEX"
            value={hnsw?.size ?? '—'}
            detail="cosine ANN"
          />
          <StatCard
            label="GIN INDEX"
            value={gin?.size ?? '—'}
            detail="full-text search"
          />
          <StatCard
            label="TOTAL STORAGE"
            value={stats?.total_size ?? '—'}
            detail="serving relation"
          />
        </section>

        <section className="panel search-panel">
          <div className="section-heading">
            <div>
              <span className="eyebrow">SEARCH</span>
              <h3>Query the corpus</h3>
            </div>

            <div className="method-tabs">
              {METHODS.map((item) => (
                <button
                  key={item.id}
                  className={method === item.id ? 'active' : ''}
                  onClick={() => setMethod(item.id)}
                  type="button"
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <form className="search-form" onSubmit={handleSearch}>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search 500,000 documents..."
              aria-label="Search query"
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </form>

          <div className="search-caption">
            {method === 'hybrid' &&
              'PostgreSQL FTS + HNSW candidates fused with RRF'}
            {method === 'vector' &&
              'MiniLM query embedding + pgvector HNSW cosine search'}
            {method === 'lexical' &&
              'PostgreSQL websearch_to_tsquery + GIN full-text index'}
          </div>
        </section>

        {result && (
          <>
            <section className="performance-grid">
              {result.embedding_ms !== undefined && (
                <StatCard
                  label="EMBEDDING"
                  value={formatMs(result.embedding_ms)}
                  detail="query encoding"
                />
              )}
              {result.lexical_ms !== undefined && (
                <StatCard
                  label="LEXICAL"
                  value={formatMs(result.lexical_ms)}
                  detail="GIN retrieval"
                />
              )}
              {result.vector_ms !== undefined && (
                <StatCard
                  label="VECTOR"
                  value={formatMs(result.vector_ms)}
                  detail="HNSW retrieval"
                />
              )}
              {result.retrieval_ms !== undefined && (
                <StatCard
                  label="RETRIEVAL"
                  value={formatMs(result.retrieval_ms)}
                  detail="database search"
                />
              )}
              {result.fusion_ms !== undefined && (
                <StatCard
                  label="FUSION"
                  value={formatMs(result.fusion_ms)}
                  detail="RRF"
                />
              )}
              <StatCard
                label="TOTAL"
                value={formatMs(result.total_ms)}
                detail={result.method}
              />
            </section>

            <section className="panel results-panel">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">RESULTS</span>
                  <h3>{result.results.length} documents returned</h3>
                </div>
                <span className="query-label">“{result.query}”</span>
              </div>

              <div className="results-list">
                {result.results.map((item, index) => (
                  <article
                    className="result-card"
                    key={`${item.document_id}-${index}`}
                  >
                    <div className="result-rank">
                      #{String(index + 1).padStart(2, '0')}
                    </div>

                    <div className="result-body">
                      <div className="result-meta">
                        <span>{item.external_id ?? `doc-${item.document_id}`}</span>

                        {item.similarity !== undefined && (
                          <span>
                            similarity {item.similarity.toFixed(4)}
                          </span>
                        )}

                        {item.score !== undefined && (
                          <span>FTS {item.score.toFixed(4)}</span>
                        )}

                        {item.rrf_score !== undefined && (
                          <span>RRF {item.rrf_score.toFixed(5)}</span>
                        )}

                        {item.lexical_rank != null && (
                          <span>lexical #{item.lexical_rank}</span>
                        )}

                        {item.vector_rank != null && (
                          <span>vector #{item.vector_rank}</span>
                        )}
                      </div>

                      <p>{item.content}</p>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}

        <DocumentManager />

        <section className="lower-grid">
          <div className="panel">
            <div className="section-heading compact">
              <div>
                <span className="eyebrow">SEARCH HISTORY</span>
                <h3>Recent queries</h3>
              </div>
            </div>

            <div className="history-list">
              {history.length === 0 && (
                <p className="empty-state">No searches recorded yet.</p>
              )}

              {history.map((item) => (
                <div className="history-row" key={item.search_id}>
                  <div>
                    <strong>{item.query_text}</strong>
                    <span>{item.search_type}</span>
                  </div>
                  <div className="history-metrics">
                    <span>{item.result_count} results</span>
                    <strong>{formatMs(item.latency_ms)}</strong>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="section-heading compact">
              <div>
                <span className="eyebrow">DATABASE</span>
                <h3>Index footprint</h3>
              </div>
            </div>

            <div className="index-list">
              {indexes.map((item) => (
                <div className="index-row" key={item.name}>
                  <div>
                    <strong>{item.name}</strong>
                    <span>
                      {item.name.includes('hnsw')
                        ? 'Approximate nearest neighbor'
                        : item.name.includes('gin')
                          ? 'Full-text inverted index'
                          : 'Relational index'}
                    </span>
                  </div>
                  <b>{item.size}</b>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer>
        Hybrid Vector Retrieval System · PostgreSQL 17 · pgvector · FastAPI
      </footer>
    </div>
  )
}

export default App
