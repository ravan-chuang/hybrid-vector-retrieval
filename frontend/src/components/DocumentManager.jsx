import { useEffect, useState } from 'react'
import { documentApi } from '../api'

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
    publication_year: form.publication_year
      ? Number(form.publication_year)
      : null,
  }
}

function DocumentManager() {
  const [documents, setDocuments] = useState([])
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

  async function refreshDocuments(preferredId = null) {
    const data = await documentApi.list()
    setDocuments(data)

    if (preferredId && data.some((doc) => doc.document_id === preferredId)) {
      setSelectedId(preferredId)
    } else if (selectedId && data.some((doc) => doc.document_id === selectedId)) {
      setSelectedId(selectedId)
    } else {
      setSelectedId(data[0]?.document_id ?? null)
    }

    return data
  }

  useEffect(() => {
    let cancelled = false

    documentApi
      .list()
      .then((data) => {
        if (cancelled) return
        setDocuments(data)
        setSelectedId(data[0]?.document_id ?? null)
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
        setForm({
          title: document.title ?? '',
          content: document.content ?? '',
          author: document.author ?? '',
          source: document.source ?? '',
          publication_year: document.publication_year ?? '',
        })
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
      setForm({
        title: selected.title ?? '',
        content: selected.content ?? '',
        author: selected.author ?? '',
        source: selected.source ?? '',
        publication_year: selected.publication_year ?? '',
      })
    } else if (documents.length > 0) {
      setSelectedId(documents[0].document_id)
    }
  }

  async function saveDocument(event) {
    event.preventDefault()

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
        await refreshDocuments(created.document_id)
        setCreating(false)
        setEditing(false)
        setMessage(`Document #${created.document_id} created.`)
      } else {
        const updated = await documentApi.update(selectedId, payload)
        await refreshDocuments(updated.document_id)
        setSelected(updated)
        setEditing(false)
        setMessage(`Document #${updated.document_id} updated.`)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function deleteDocument() {
    if (!selectedId || !selected) return

    const confirmed = window.confirm(
      `Delete "${selected.title}"?\n\nIts document_chunks and document_tags associations will be removed by the database cascade rules.`,
    )

    if (!confirmed) return

    setSaving(true)
    setError('')
    setMessage('')

    try {
      await documentApi.remove(selectedId)
      setSelected(null)
      setTags([])
      setEditing(false)
      await refreshDocuments()
      setMessage('Document deleted. Foreign-key cascade completed.')
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function addTag(event) {
    event.preventDefault()

    const cleanName = tagName.trim()
    if (!cleanName || !selectedId) return

    setSaving(true)
    setError('')

    try {
      await documentApi.addTag(selectedId, cleanName)
      setTags(await documentApi.tags(selectedId))
      setTagName('')
      setMessage(`Tag "${cleanName}" associated with this document.`)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function removeTag(tag) {
    if (!selectedId) return

    setSaving(true)
    setError('')

    try {
      await documentApi.removeTag(selectedId, tag.tag_id)
      setTags(await documentApi.tags(selectedId))
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
          disabled={saving}
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
                onClick={() => {
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
          {(selected || creating) && (
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
                      onClick={() => setEditing(true)}
                    >
                      Edit
                    </button>
                    <button
                      className="danger-button"
                      type="button"
                      onClick={deleteDocument}
                      disabled={saving}
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
                      disabled={saving}
                    >
                      Cancel
                    </button>
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={saving}
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
                        <strong>{selected.publication_year || '—'}</strong>
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
                              disabled={saving}
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
                          disabled={saving || !tagName.trim()}
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

          {!selected && !creating && (
            <p className="empty-state">
              Select a document or create a new one.
            </p>
          )}
        </div>
      </div>
    </section>
  )
}

export default DocumentManager
