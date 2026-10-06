const API_BASE = 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let message = `HTTP ${response.status}`
    try {
      const body = await response.json()
      message = body.detail || message
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message)
  }

  return response.json()
}

export const api = {
  health: () => request('/health'),
  stats: () => request('/stats'),

  vectorSearch: (query, topK = 10, efSearch = 40) =>
    request('/search/vector', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: topK,
        ef_search: efSearch,
      }),
    }),

  lexicalSearch: (query, topK = 10) =>
    request('/search/lexical', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: topK,
      }),
    }),

  hybridSearch: (
    query,
    topK = 10,
    candidateK = 50,
    efSearch = 40,
    rrfK = 60,
  ) =>
    request('/search/hybrid', {
      method: 'POST',
      body: JSON.stringify({
        query,
        top_k: topK,
        candidate_k: candidateK,
        ef_search: efSearch,
        rrf_k: rrfK,
      }),
    }),

  history: (limit = 8) => request(`/search/history?limit=${limit}`),
}

export const documentApi = {
  list: () => request('/documents'),

  get: (documentId) =>
    request(`/documents/${documentId}`),

  create: (document) =>
    request('/documents', {
      method: 'POST',
      body: JSON.stringify(document),
    }),

  update: (documentId, document) =>
    request(`/documents/${documentId}`, {
      method: 'PUT',
      body: JSON.stringify(document),
    }),

  remove: (documentId) =>
    request(`/documents/${documentId}`, {
      method: 'DELETE',
    }),

  tags: (documentId) =>
    request(`/documents/${documentId}/tags`),

  addTag: (documentId, name) =>
    request(`/documents/${documentId}/tags`, {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  removeTag: (documentId, tagId) =>
    request(`/documents/${documentId}/tags/${tagId}`, {
      method: 'DELETE',
    }),
}
