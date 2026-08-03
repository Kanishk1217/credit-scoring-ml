import type { InputHTMLAttributes } from 'react'
import { useState } from 'react'
import { ArrowUpRight } from 'lucide-react'

type Kind = 'access' | 'talk'
type Stage = 'idle' | 'form' | 'sending' | 'sent' | 'error'

const CONTACT_EMAIL = 'preet.d@agilitytech.ai'

function Field({ label, ...props }: { label: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="flex flex-col gap-1.5 text-left">
      <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">{label}</span>
      <input
        {...props}
        className="border-b border-line bg-transparent pb-2 font-display text-lg text-ink outline-none transition-colors focus:border-ink placeholder:font-body placeholder:text-base placeholder:text-faint"
      />
    </label>
  )
}

export default function RequestAccessForm() {
  const [stage, setStage] = useState<Stage>('idle')
  const [kind, setKind] = useState<Kind>('access')
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [org, setOrg] = useState('')
  const [details, setDetails] = useState('')
  const [website, setWebsite] = useState('') // honeypot, must stay empty

  const valid = name.trim().length > 1 && /\S+@\S+\.\S+/.test(email)

  function open(next: Kind) {
    setKind(next)
    setStage('form')
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!valid || stage === 'sending') return
    setStage('sending')
    try {
      const resp = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, org, details, kind, website }),
      })
      if (!resp.ok) throw new Error('send failed')
      setStage('sent')
    } catch {
      setStage('error')
    }
  }

  if (stage === 'idle') {
    return (
      <div className="reveal mt-12 flex flex-wrap justify-center gap-8">
        <button
          type="button"
          onClick={() => open('access')}
          className="group inline-flex cursor-pointer items-center gap-2 border-b border-ink pb-1 font-display text-lg font-500 text-ink"
        >
          Request access
          <ArrowUpRight className="h-5 w-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </button>
        <button
          type="button"
          onClick={() => open('talk')}
          className="cursor-pointer text-lg text-muted transition-colors hover:text-ink"
        >
          Talk to us
        </button>
      </div>
    )
  }

  if (stage === 'sent') {
    return (
      <div className="reveal mx-auto mt-12 max-w-md text-left">
        <p className="font-display text-xl text-ink">Thanks, {name.split(' ')[0]}.</p>
        <p className="mt-3 leading-relaxed text-muted">
          That's on its way to <span className="text-ink">{CONTACT_EMAIL}</span> — we'll reply to{' '}
          <span className="text-ink">{email}</span> directly.
        </p>
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="reveal mx-auto mt-12 grid max-w-md gap-6">
      <h3 className="font-display text-lg text-ink">
        {kind === 'access' ? 'Tell us about your lending business' : 'What do you want to ask?'}
      </h3>

      {/* honeypot -- hidden from real visitors, bots tend to fill every field */}
      <input
        type="text" value={website} onChange={(e) => setWebsite(e.target.value)} tabIndex={-1}
        autoComplete="off" aria-hidden="true"
        style={{ position: 'absolute', left: '-9999px', width: 1, height: 1, opacity: 0 }}
      />

      <Field label="Name" required value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" />
      <Field
        label="Work email" type="email" required value={email}
        onChange={(e) => setEmail(e.target.value)} placeholder="you@lender.com"
      />
      {kind === 'access' && (
        <Field label="Organization" value={org} onChange={(e) => setOrg(e.target.value)} placeholder="Optional" />
      )}
      <label className="flex flex-col gap-1.5 text-left">
        <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">
          {kind === 'access' ? 'Tell us about your data' : 'Your message'}
        </span>
        <textarea
          value={details}
          onChange={(e) => setDetails(e.target.value)}
          rows={3}
          placeholder={kind === 'access'
            ? "Applicant volume, current default rate, what you use for scoring today -- whatever's useful."
            : 'A couple lines is plenty.'}
          className="resize-none border-b border-line bg-transparent pb-2 text-ink outline-none transition-colors focus:border-ink placeholder:text-faint"
        />
      </label>

      {stage === 'error' && (
        <p className="text-sm text-[var(--color-danger)]">Couldn't send that — check your connection and try again.</p>
      )}

      <div className="flex items-center gap-6">
        <button
          type="submit"
          disabled={!valid || stage === 'sending'}
          className="inline-flex cursor-pointer items-center gap-2 border-b border-ink pb-1 font-display text-lg font-500 text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          {stage === 'sending' ? 'Sending…' : 'Send'} <ArrowUpRight className="h-5 w-5" />
        </button>
        <button type="button" onClick={() => setStage('idle')} className="cursor-pointer text-sm text-muted hover:text-ink">
          Cancel
        </button>
      </div>
    </form>
  )
}
