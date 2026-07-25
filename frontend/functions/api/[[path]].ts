// Cloudflare Pages Function: proxies /api/* to the backend and injects the API key
// server-side, so the browser never holds it. Mirrors the Vite dev proxy.
// Set BACKEND_URL and API_KEY in the Cloudflare Pages project (API_KEY as a secret).

export const onRequest = async (context: any) => {
  const { request, env } = context
  const url = new URL(request.url)
  const path = url.pathname.replace(/^\/api/, '') || '/'
  const backend = String(env.BACKEND_URL || '').replace(/\/$/, '')

  if (!backend) return new Response(JSON.stringify({ detail: 'backend not configured' }), { status: 503 })

  const headers = new Headers(request.headers)
  headers.set('X-API-Key', String(env.API_KEY || ''))
  headers.delete('host')

  const method = request.method
  const resp = await fetch(backend + path + url.search, {
    method,
    headers,
    body: method === 'GET' || method === 'HEAD' ? undefined : request.body,
  })

  const out = new Headers()
  out.set('content-type', resp.headers.get('content-type') || 'application/json')
  return new Response(resp.body, { status: resp.status, headers: out })
}
