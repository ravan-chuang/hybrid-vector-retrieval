import assert from 'node:assert/strict'
import test from 'node:test'
import { JSDOM } from 'jsdom'
import { createServer } from 'vite'
import { act, createElement } from 'react'

test('changing search settings ignores responses from the previous selection', async () => {
  const dom = new JSDOM('<main></main>')
  globalThis.window = dom.window
  globalThis.document = dom.window.document
  globalThis.IS_REACT_ACT_ENVIRONMENT = true

  const { createRoot } = await import('react-dom/client')
  const vite = await createServer({
    configFile: false,
    server: { middlewareMode: true, ws: false },
    appType: 'custom',
  })
  const root = createRoot(document.querySelector('main'))
  const previousFetch = globalThis.fetch
  const requests = []
  globalThis.fetch = async (url, options) => {
    const path = new URL(url).pathname
    if (path === '/stats') return Response.json({ documents: 100 })
    if (path === '/health') return Response.json({ database_check: true })
    if (path === '/documents' || path === '/search/history') return Response.json([])
    assert.ok(path.startsWith('/search/'), path)
    return new Promise((resolve) => {
      requests.push({ path, body: JSON.parse(options.body), resolve })
    })
  }

  function button(label) {
    return [...document.querySelectorAll('button')]
      .find((element) => element.textContent.trim() === label)
  }
  const search = () => button('Search').click()
  const successful = (content) => Response.json({
    query: 'Manhattan Project',
    results: [{ document_id: 1, content }],
  })

  try {
    const { default: App } = await vite.ssrLoadModule('/src/App.jsx')
    await act(async () => root.render(createElement(App)))
    await act(async () => search())

    await act(async () => button('Vector').click())
    await act(async () => search())
    assert.deepEqual(requests.map(({ path }) => path), ['/search/hybrid', '/search/vector'])

    await act(async () => requests[0].resolve(successful('Obsolete hybrid result')))
    assert.ok(button('Searching…').disabled)
    assert.equal(document.querySelector('.results-panel'), null)

    const corpus = document.querySelector('#search-corpus')
    await act(async () => {
      corpus.value = 'application'
      corpus.dispatchEvent(new window.Event('change', { bubbles: true }))
    })
    await act(async () => search())
    assert.equal(requests[1].body.corpus, 'benchmark')
    assert.equal(requests[2].body.corpus, 'application')

    await act(async () => requests[2].resolve(successful('Current application result')))
    await act(async () => requests[1].resolve(Response.json({ detail: 'Old failure' }, { status: 503 })))
    assert.match(document.querySelector('.results-panel').textContent, /Current application result/)
    assert.doesNotMatch(document.body.textContent, /Obsolete hybrid result|Old failure/)
    assert.ok(!button('Search').disabled)

    await act(async () => button('Hybrid').click())
    await act(async () => search())
    const fusion = document.querySelector('#fusion-strategy')
    await act(async () => {
      fusion.value = 'balanced'
      fusion.dispatchEvent(new window.Event('change', { bubbles: true }))
    })
    await act(async () => search())
    assert.equal(requests[3].body.lexical_weight, 0.1)
    assert.equal(requests[4].body.lexical_weight, 0.5)

    await act(async () => requests[4].resolve(successful('Current balanced result')))
    await act(async () => requests[3].resolve(successful('Obsolete dense-weighted result')))
    assert.match(document.querySelector('.results-panel').textContent, /Current balanced result/)
    assert.doesNotMatch(document.body.textContent, /Obsolete dense-weighted result/)
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
