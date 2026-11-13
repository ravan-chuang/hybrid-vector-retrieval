import assert from 'node:assert/strict'
import test from 'node:test'
import { JSDOM } from 'jsdom'
import { createServer } from 'vite'
import { act, createElement } from 'react'

test('document mutations follow the loaded details when selection changes', async () => {
  const dom = new JSDOM('<main></main>')
  globalThis.window = dom.window
  globalThis.document = dom.window.document
  globalThis.IS_REACT_ACT_ENVIRONMENT = true

  const { createRoot } = await import('react-dom/client')
  const vite = await createServer({
    configFile: false,
    server: { middlewareMode: true, hmr: false },
    appType: 'custom',
  })
  const root = createRoot(document.querySelector('main'))
  const previousFetch = globalThis.fetch
  const documents = [
    { document_id: 1, title: 'Document A', content: 'First', author: 'Alice', publication_year: 0 },
    { document_id: 2, title: 'Document B', content: 'Second' },
    { document_id: 3, title: 'Document C', content: 'Third' },
  ]
  const deletions = []
  const updates = []
  let confirmation
  let finishB
  const pendingB = new Promise((resolve) => { finishB = resolve })

  window.confirm = (prompt) => {
    confirmation = prompt
    return true
  }
  globalThis.fetch = async (url, options) => {
    const path = new URL(url).pathname
    if (options?.method === 'DELETE') {
      deletions.push(path)
      documents.splice(documents.findIndex((doc) => path === `/documents/${doc.document_id}`), 1)
      return Response.json({ deleted: true })
    }
    if (options?.method === 'PUT') {
      const changes = JSON.parse(options.body)
      updates.push(changes)
      Object.assign(documents[0], changes)
      return Response.json(documents[0])
    }
    if (path === '/documents') return Response.json(documents)
    if (path.endsWith('/tags')) return Response.json([])
    if (path === '/documents/2') return pendingB
    if (path === '/documents/3') return Response.json({ detail: 'Unavailable' }, { status: 503 })
    return Response.json(documents[0])
  }

  function button(label) {
    return [...document.querySelectorAll('button')]
      .find((element) => element.textContent.trim() === label)
  }
  const selectDocument = (index) => document.querySelectorAll('.document-nav')[index].click()

  try {
    const { default: DocumentManager } = await vite.ssrLoadModule('/src/components/DocumentManager.jsx')
    await act(async () => root.render(createElement(DocumentManager)))
    assert.ok(button('Delete'))
    await act(async () => button('Edit').click())
    const author = document.querySelector('[name="author"]')
    const setValue = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
    await act(async () => {
      setValue.call(author, '')
      author.dispatchEvent(new window.Event('input', { bubbles: true }))
    })
    await act(async () => button('Save changes').click())
    assert.deepEqual(updates, [{ author: null }])
    assert.equal(documents[0].publication_year, 0)

    await act(async () => selectDocument(1))
    assert.match(document.body.textContent, /Loading document/)
    assert.equal(button('Edit'), undefined)
    assert.equal(button('Delete'), undefined)
    assert.equal(button('Add tag'), undefined)
    assert.deepEqual(deletions, [])

    await act(async () => finishB(Response.json(documents[1])))
    await act(async () => button('Delete').click())
    assert.match(confirmation, /Delete "Document B"/)
    assert.deepEqual(deletions, ['/documents/2'])

    await act(async () => selectDocument(1))
    assert.match(document.body.textContent, /Unavailable/)
    assert.equal(button('Delete'), undefined)
  } finally {
    await act(async () => root.unmount())
    await vite.close()
    dom.window.close()
    globalThis.fetch = previousFetch
    delete globalThis.window
    delete globalThis.document
    delete globalThis.IS_REACT_ACT_ENVIRONMENT
  }
})
