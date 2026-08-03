import type { InputHTMLAttributes } from 'react'
import { useState } from 'react'
import { ArrowUpRight } from 'lucide-react'

/* No backend to receive leads yet, so "send" opens a prefilled mailto: to the team inbox --
   real delivery, zero infra. Swap buildMailto's call site for a fetch() once there's an endpoint. */
const CONTACT_EMAIL = 'preet.d@agilitytech.ai'
const TALK_MAILTO = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent('Quick question about CreditScore')}`

type Stage = 'idle' | 'form' | 'sent'

function buildMailto(name: string, email: string, org: string, details: string): string {
  const subject = `Access request${org ? ` -- ${org}` : ''}`
  const body = [
    `Name: ${name}`,
    `Email: ${email}`,
    org ? `Organization: ${org}` : null,
    '',
    'About their data / use case:',
    details || '(not provided)',
  ].filter((l) => l !== null).join('\n')
  return `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`
}

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
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [org, setOrg] = useState('')
  const [details, setDetails] = useState('')

  const valid = name.trim().length > 1 && /\S+@\S+\.\S+/.test(email)
  const mailto = buildMailto(name, email, org, details)

  function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!valid) return
    window.location.href = mailto
    setStage('sent')
  }

  if (stage === 'idle') {
    return (
      <div className="reveal mt-12 flex flex-wrap justify-center gap-8">
        <button
          type="button"
          onClick={() => setStage('form')}
          className="group inline-flex cursor-pointer items-center gap-2 border-b border-ink pb-1 font-display text-lg font-500 text-ink"
        >
          Request access
          <ArrowUpRight className="h-5 w-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </button>
        <a href={TALK_MAILTO} className="text-lg text-muted transition-colors hover:text-ink">Talk to us</a>
      </div>
    )
  }

  if (stage === 'sent') {
    return (
      <div className="reveal mx-auto mt-12 max-w-md text-left">
        <p className="font-display text-xl text-ink">Thanks, {name.split(' ')[0]}.</p>
        <p className="mt-3 leading-relaxed text-muted">
          We opened an email to <span className="text-ink">{CONTACT_EMAIL}</span> with your details filled
          in — send it from your mail app and we'll reply to <span className="text-ink">{email}</span> directly.
        </p>
        <a href={mailto} className="mt-4 inline-block border-b border-ink text-sm text-ink">
          Didn't open? Send it manually
        </a>
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="reveal mx-auto mt-12 grid max-w-md gap-6">
      <Field label="Name" required value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" />
      <Field
        label="Work email" type="email" required value={email}
        onChange={(e) => setEmail(e.target.value)} placeholder="you@lender.com"
      />
      <Field label="Organization" value={org} onChange={(e) => setOrg(e.target.value)} placeholder="Optional" />
      <label className="flex flex-col gap-1.5 text-left">
        <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">Tell us about your data</span>
        <textarea
          value={details}
          onChange={(e) => setDetails(e.target.value)}
          rows={3}
          placeholder="Applicant volume, current default rate, what you use for scoring today -- whatever's useful."
          className="resize-none border-b border-line bg-transparent pb-2 text-ink outline-none transition-colors focus:border-ink placeholder:text-faint"
        />
      </label>
      <div className="flex items-center gap-6">
        <button
          type="submit"
          disabled={!valid}
          className="inline-flex cursor-pointer items-center gap-2 border-b border-ink pb-1 font-display text-lg font-500 text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          Send request <ArrowUpRight className="h-5 w-5" />
        </button>
        <button type="button" onClick={() => setStage('idle')} className="cursor-pointer text-sm text-muted hover:text-ink">
          Cancel
        </button>
      </div>
    </form>
  )
}
