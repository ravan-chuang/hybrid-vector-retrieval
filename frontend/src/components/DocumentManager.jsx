import { useEffect, useState } from 'react'
import { documentApi } from '../api'

const PAGE_SIZE = 20

const EMPTY_FORM = {
  title: '',
  content: '',
  author: '',
  source: '',
  publication_year: '',
}

function toPayload(form) {
  return {
    title: form.title.trim(),
    content: form.content.trim(),
    author: form.author.trim() || null,
    source: form.source.trim() || null,
    publication_year: form.publication_year === ''
      ? null
      : Number(form.publication_year),
  }
}

function documentToForm(document) {
  return {
    title: document.title ?? '',
    content: document.content ?? '',
    author: document.author ?? '',
    source: document.source ?? '',
    publication_year: document.publication_year ?? '',
  }
}

async function fetchDocumentPage(index) {
  const rows = await documentApi.list(PAGE_SIZE + 1, index * PAGE_SIZE)
  return { index, documents: rows.slice(0, PAGE_SIZE), hasNext: rows.length > PAGE_SIZE }
}

function DocumentManager() {
  const [page, setPage] = useState({ index: 0, documents: [], hasNext: false })
  const { documents } = page
  const [selectedId, setSelectedId] = useState(null)
  const [selected, setSelected] = useState(null)
  const [tags, setTags] = useState([])
  const [form, setForm] = useState(EMPTY_FORM)
  const [tagName, setTagName] = useState('')
  const [editing, setEditing] = useState(false)
  const [creating, setCreating] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const hasSelectedDocument = selected !== null && selected.document_id === selectedId

  async function refreshDocuments(index, preferredId = null) {
    let nextPage = await fetchDocumentPage(index)
    while (nextPage.documents.length === 0 && nextPage.index > 0) {
      nextPage = await fetchDocumentPage(nextPage.index - 1)
    }

    setPage(nextPage)
    setSelectedId((current) => {
      const target = preferredId ?? current
      return nextPage.documents.some((doc) => doc.document_id === target)
        ? target
        : nextPage.documents[0]?.document_id ?? null
    })
  }

  async function changePage(index) {
    setLoading(true)
    setError('')
    setMessage('')
    try {
      await refreshDocuments(index)
      setTagName('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let cancelled = false

    fetchDocumentPage(0)
      .then((firstPage) => {
        if (cancelled) return
        setPage(firstPage)
        setSelectedId(firstPage.documents[0]?.document_id ?? null)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedId || creating) {
      return undefined
    }

    let cancelled = false

    Promise.all([
      documentApi.get(selectedId),
      documentApi.tags(selectedId),
    ])
      .then(([document, documentTags]) => {
        if (cancelled) return

        setSelected(document)
        setTags(documentTags)
        setForm(documentToForm(document))
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })

    return () => {
      cancelled = true
    }
  }, [selectedId, creating])

  function updateField(event) {
    const { name, value } = event.target
    setForm((current) => ({
      ...current,
      [name]: value,
    }))
  }

  function beginCreate() {
    setCreating(true)
    setEditing(true)
    setSelected(null)
    setSelectedId(null)
    setTags([])
    setForm(EMPTY_FORM)
    setError('')
    setMessage('')
  }

  function cancelEdit() {
    setCreating(false)
    setEditing(false)
    setError('')
    setMessage('')

    if (selected) {
      setForm(documentToForm(selected))
    } else if (documents.length > 0) {
      setSelectedId(documents[0].document_id)
    }
  }

  async function saveDocument(event) {
    event.preventDefault()

    if (saving || loading || (!creating && !hasSelectedDocument)) return

    if (!form.title.trim() || !form.content.trim()) {
      setError('Title and content are required.')
      return
    }

    setSaving(true)
    setError('')
    setMessage('')

    try {
      const payload = toPayload(form)

      if (creating) {
        const created = await documentApi.create(payload)
        setCreating(false)
        setEditing(false)
        setMessage(`Document #${created.document_id} created.`)
        await refreshDocuments(0, created.document_id)
      } else {
        const original = toPayload(documentToForm(selected))
        const changes = Object.fromEntries(
          Object.entries(payload).filter(([field, value]) => value !== original[field]),
        )
        const updated = await documentApi.update(selected.document_id, changes)
        setSelected(updated)
        setForm(documentToForm(updated))
        setEditing(false)
        setMessage(`Document #${updated.document_id} updated.`)
        await refreshDocuments(page.index, updated.document_id)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function deleteDocument() {
    if (saving || loading || !hasSelectedDocument) return

    const confirmed = window.confirm(
      `Delete "${selected.title}"?\n\nIts document_chunks and document_tags associations will be removed by the database cascade rules.`,
    )

    if (!confirmed) return

    setSaving(true)
    setError('')
    setMessage('')

    try {
      await documentApi.remove(selected.document_id)
      setSelected(null)
      setSelectedId(null)
      setTags([])
      setEditing(false)
      setMessage('Document deleted. Foreign-key cascade completed.')
      await refreshDocuments(page.index)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function addTag(event) {
    event.preventDefault()

    const cleanName = tagName.trim()
    if (saving || loading || !cleanName || !hasSelectedDocument) return

    setSaving(true)
    setError('')

    try {
      await documentApi.addTag(selected.document_id, cleanName)
      setTags(await documentApi.tags(selected.document_id))
      setTagName('')
      setMessage(`Tag "${cleanName}" associated with this document.`)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function removeTag(tag) {
    if (saving || loading || !hasSelectedDocument) return

    setSaving(true)
    setError('')

    try {
      await documentApi.removeTag(selected.document_id, tag.tag_id)
      setTags(await documentApi.tags(selected.document_id))
      setMessage(
        `Association with "${tag.name}" removed. The tag itself remains in the tags table.`,
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="panel document-manager">
      <div className="section-heading">
        <div>
          <span className="eyebrow">RELATIONAL DATABASE</span>
          <h3>Documents &amp; Tags</h3>
        </div>

        <button
          className="secondary-button"
          type="button"
          onClick={beginCreate}
          disabled={saving || loading}
        >
          + New document
        </button>
      </div>

      <div className="relation-strip">
        <span>documents</span>
        <b>1</b>
        <i>↔</i>
        <b>N</b>
        <span>document_tags</span>
        <b>N</b>
        <i>↔</i>
        <b>1</b>
        <span>tags</span>
      </div>

      {error && <div className="manager-alert error">{error}</div>}
      {message && <div className="manager-alert success">{message}</div>}

      <div className="document-layout">
        <aside className="document-sidebar">
          <div className="sidebar-title">
            <span>DOCUMENTS</span>
            <b>{documents.length}</b>
          </div>

          {(page.index > 0 || page.hasNext) && (
            <nav className="document-pagination" aria-label="Document pages">
              <button
                className="secondary-button"
                type="button"
                disabled={loading || saving || editing || page.index === 0}
                onClick={() => changePage(page.index - 1)}
              >
                Previous
              </button>
              <span aria-live="polite">Page {page.index + 1}</span>
              <button
                className="secondary-button"
                type="button"
                disabled={loading || saving || editing || !page.hasNext}
                onClick={() => changePage(page.index + 1)}
              >
                Next
              </button>
            </nav>
          )}

          {loading && <p className="empty-state">Loading documents…</p>}

          {!loading &&
            documents.map((document) => (
              <button
                key={document.document_id}
                type="button"
                className={
                  selectedId === document.document_id && !creating
                    ? 'document-nav active'
                    : 'document-nav'
                }
                disabled={saving || loading}
                onClick={() => {
                  if (selectedId !== document.document_id) {
                    setSelected(null)
                    setTags([])
                    setTagName('')
                  }
                  setCreating(false)
                  setEditing(false)
                  setSelectedId(document.document_id)
                  setError('')
                  setMessage('')
                }}
              >
                <span>#{document.document_id}</span>
                <strong>{document.title}</strong>
                <small>{document.author || 'Unknown author'}</small>
              </button>
            ))}
        </aside>

        <div className="document-workspace">
          {(hasSelectedDocument || creating) && (
            <>
              <div className="document-toolbar">
                <div>
                  <span className="eyebrow">
                    {creating
                      ? 'CREATE'
                      : `DOCUMENT #${selected?.document_id ?? ''}`}
                  </span>
                  <h4>
                    {creating
                      ? 'New document'
                      : selected?.title}
                  </h4>
                </div>

                {!creating && !editing && (
                  <div className="toolbar-actions">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => {
                        setForm(documentToForm(selected))
                        setEditing(true)
                      }}
                      disabled={saving || loading}
                    >
                      Edit
                    </button>
                    <button
                      className="danger-button"
                      type="button"
                      onClick={deleteDocument}
                      disabled={saving || loading}
                    >
                      Delete
                    </button>
                  </div>
                )}
              </div>

              {editing ? (
                <form className="document-form" onSubmit={saveDocument}>
                  <label>
                    Title
                    <input
                      name="title"
                      value={form.title}
                      onChange={updateField}
                      maxLength="500"
                      required
                    />
                  </label>

                  <label className="full-field">
                    Content
                    <textarea
                      name="content"
                      value={form.content}
                      onChange={updateField}
                      rows="6"
                      required
                    />
                  </label>

                  <label>
                    Author
                    <input
                      name="author"
                      value={form.author}
                      onChange={updateField}
                      maxLength="255"
                    />
                  </label>

                  <label>
                    Publication year
                    <input
                      name="publication_year"
                      type="number"
                      min="0"
                      max="2100"
                      value={form.publication_year}
                      onChange={updateField}
                    />
                  </label>

                  <label className="full-field">
                    Source
                    <input
                      name="source"
                      value={form.source}
                      onChange={updateField}
                      maxLength="500"
                    />
                  </label>

                  <div className="form-actions full-field">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={cancelEdit}
                      disabled={saving || loading}
                    >
                      Cancel
                    </button>
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={saving || loading}
                    >
                      {saving
                        ? 'Saving…'
                        : creating
                          ? 'Create document'
                          : 'Save changes'}
                    </button>
                  </div>
                </form>
              ) : (
                selected && (
                  <>
                    <div className="document-details">
                      <div>
                        <span>AUTHOR</span>
                        <strong>{selected.author || '—'}</strong>
                      </div>
                      <div>
                        <span>YEAR</span>
                        <strong>{selected.publication_year ?? '—'}</strong>
                      </div>
                      <div>
                        <span>SOURCE</span>
                        <strong>{selected.source || '—'}</strong>
                      </div>
                      <div>
                        <span>CREATED</span>
                        <strong>
                          {new Date(selected.created_at).toLocaleString()}
                        </strong>
                      </div>
                    </div>

                    <div className="document-content">
                      {selected.content}
                    </div>

                    <div className="tag-section">
                      <div className="tag-heading">
                        <div>
                          <span className="eyebrow">M:N RELATIONSHIP</span>
                          <h4>Tags</h4>
                        </div>

                        <span className="tag-count">
                          {tags.length} associated
                        </span>
                      </div>

                      <div className="tag-list">
                        {tags.map((tag) => (
                          <span className="tag-chip" key={tag.tag_id}>
                            {tag.name}
                            <button
                              type="button"
                              aria-label={`Remove ${tag.name}`}
                              onClick={() => removeTag(tag)}
                              disabled={saving || loading}
                            >
                              ×
                            </button>
                          </span>
                        ))}

                        {tags.length === 0 && (
                          <span className="empty-state">
                            No tags associated.
                          </span>
                        )}
                      </div>

                      <form className="tag-form" onSubmit={addTag}>
                        <input
                          value={tagName}
                          onChange={(event) => setTagName(event.target.value)}
                          placeholder="e.g. vector-search"
                          maxLength="100"
                        />
                        <button
                          className="secondary-button"
                          type="submit"
                          disabled={saving || loading || !tagName.trim()}
                        >
                          Add tag
                        </button>
                      </form>

                      <p className="relation-note">
                        Removing a chip deletes only the row in
                        <code> document_tags </code>
                        — the referenced tag remains in
                        <code> tags</code>.
                      </p>
                    </div>
                  </>
                )
              )}
            </>
          )}

          {!hasSelectedDocument && !creating && (
            <p className="empty-state">
              {selectedId && !error
                ? 'Loading document…'
                : 'Select a document or create a new one.'}
            </p>
          )}
        </div>
      </div>
    </section>
  )
}

export default DocumentManager
