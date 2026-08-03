// Cloudflare Pages Function: receives the homepage contact/request-access form and sends a real
// email via Resend, server-side. Resolves before the [[path]].ts catch-all proxy, so this never
// reaches the Render backend. Needs RESEND_API_KEY set as a secret:
//   npx wrangler pages secret put RESEND_API_KEY --project-name=credit-scoring-ml
//
// TO_ADDRESS must match the email the Resend account was signed up with (zesus680@gmail.com) --
// Resend's sandbox sender (onboarding@resend.dev, no verified domain) refuses any other
// recipient. Switch this to a business address once a domain is verified in Resend.
const TO_ADDRESS = 'zesus680@gmail.com'
const FROM_ADDRESS = 'CreditScore <onboarding@resend.dev>'

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

export const onRequestPost = async (context: any) => {
  const { request, env } = context

  let body: any
  try {
    body = await request.json()
  } catch {
    return json(400, { error: 'invalid request body' })
  }

  const { name, email, org, details, kind, website } = body || {}

  // Honeypot: a real visitor never fills a field named "website" that's hidden via CSS.
  if (typeof website === 'string' && website.trim().length > 0) {
    return json(200, { ok: true })
  }
  if (typeof name !== 'string' || name.trim().length < 2) {
    return json(400, { error: 'name is required' })
  }
  if (typeof email !== 'string' || !/\S+@\S+\.\S+/.test(email)) {
    return json(400, { error: 'a valid email is required' })
  }

  const apiKey = env.RESEND_API_KEY
  if (!apiKey) {
    return json(503, { error: 'email delivery is not configured yet' })
  }

  const subject = kind === 'talk'
    ? `Quick question from ${name}`
    : `Access request${org ? ` -- ${org}` : ''}`

  const text = [
    `Name: ${name}`,
    `Email: ${email}`,
    org ? `Organization: ${org}` : null,
    '',
    details ? 'Message / about their data:' : null,
    details || null,
  ].filter((line) => line !== null && line !== undefined).join('\n')

  const resendResp = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      from: FROM_ADDRESS,
      to: [TO_ADDRESS],
      reply_to: email,
      subject,
      text,
    }),
  })

  if (!resendResp.ok) {
    console.error('resend send failed', resendResp.status, await resendResp.text().catch(() => ''))
    return json(502, { error: 'failed to send -- try again in a moment' })
  }

  return json(200, { ok: true })
}
