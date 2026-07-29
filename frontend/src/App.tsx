import type { ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'
import { ArrowUpRight } from 'lucide-react'
import LiveDemo from './LiveDemo'

/* thin scroll-progress line at the very top */
function ScrollProgress() {
  const [p, setP] = useState(0)
  useEffect(() => {
    const on = () => {
      const h = document.documentElement
      setP(h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight))
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return (
    <div className="fixed inset-x-0 top-0 z-50 h-[2px]">
      <div className="h-full bg-accent transition-[width] duration-75" style={{ width: `${p * 100}%` }} />
    </div>
  )
}

/* big statement whose words fill from faint -> ink as it scrolls through the viewport */
function ScrollStatement() {
  const ref = useRef<HTMLParagraphElement>(null)
  const [p, setP] = useState(0)
  useEffect(() => {
    const on = () => {
      const el = ref.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const vh = window.innerHeight
      const prog = (vh * 0.82 - r.top) / (r.height + vh * 0.15)
      setP(Math.max(0, Math.min(1, prog)))
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  const text =
    "Credit teams don't need another black box. They need a score they can explain, defend, and audit — for every applicant, every time."
  const words = text.split(' ')
  return (
    <section className="border-t border-line bg-paper">
      <div className="mx-auto max-w-[1080px] px-6 py-28 md:px-10 md:py-44">
        <p ref={ref} className="font-display font-500 leading-[1.28] tracking-tight"
           style={{ fontSize: 'clamp(1.75rem, 4.2vw, 3.25rem)' }}>
          {words.map((w, i) => (
            <span key={i} style={{ color: p * words.length > i ? 'var(--color-ink)' : 'var(--color-faint)', transition: 'color 0.2s ease' }}>
              {w === '—' ? <span className="text-accent">{w}</span> : w}{' '}
            </span>
          ))}
        </p>
      </div>
    </section>
  )
}

function Label({ children }: { children: ReactNode }) {
  return (
    <span className="font-label text-xs font-500 uppercase tracking-[0.2em] text-faint">{children}</span>
  )
}

const CYCLE = ['legible', 'auditable', 'defensible', 'fair']
function Cycle() {
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setI((x) => (x + 1) % CYCLE.length), 2400)
    return () => clearInterval(t)
  }, [])
  return <span key={i} className="word-in italic font-500 text-accent">{CYCLE[i]}</span>
}

function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1240px] items-center justify-between px-6 py-4 md:px-10">
        <span className="font-display text-base font-600 tracking-tight text-ink">creditscore</span>
        <nav className="hidden items-center gap-10 text-sm text-muted md:flex">
          <a href="#how" className="transition-colors hover:text-ink">how it works</a>
          <a href="#caps" className="transition-colors hover:text-ink">capabilities</a>
          <a href="#output" className="transition-colors hover:text-ink">the output</a>
        </nav>
        <a href="#cta" className="group inline-flex items-center gap-1 text-sm font-500 text-ink">
          get access
          <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </a>
      </div>
    </header>
  )
}

function HeroWave() {
  let d = 'M0 100'
  for (let x = 0; x < 2400; x += 400) d += ` Q ${x + 100} 44 ${x + 200} 100 Q ${x + 300} 156 ${x + 400} 100`
  return (
    <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden" aria-hidden="true">
      <svg className="animate-wave absolute left-0 top-[54%] h-56 w-[200%]" viewBox="0 0 2400 200" preserveAspectRatio="none" fill="none">
        <path d={d} stroke="var(--color-accent)" strokeWidth="2" opacity="0.22" />
      </svg>
      <svg className="animate-wave-slow absolute left-0 top-[60%] h-56 w-[200%]" viewBox="0 0 2400 200" preserveAspectRatio="none" fill="none">
        <path d={d} stroke="var(--color-ink)" strokeWidth="1.5" opacity="0.09" />
      </svg>
    </div>
  )
}

function Hero() {
  return (
    <section className="relative overflow-hidden">
      <HeroWave />
      <div className="relative z-10 mx-auto max-w-[1240px] px-6 pt-16 md:px-10 md:pt-24">
      <div className="rise"><Label>001 &nbsp;/&nbsp; credit risk, scored</Label></div>
      <h1 className="mt-6 font-display font-700 leading-[0.94] tracking-[-0.03em] text-ink"
          style={{ fontSize: 'clamp(2.75rem, 8.5vw, 7.5rem)' }}>
        <span className="line-mask"><span className="line-rise" style={{ animationDelay: '140ms' }}>Default risk,</span></span>
        <span className="line-mask"><span className="line-rise" style={{ animationDelay: '270ms' }}>made&nbsp;<Cycle />.</span></span>
      </h1>

      <div className="mt-14 grid gap-12 md:mt-20 md:grid-cols-[1fr_1.05fr] md:items-end">
        <div className="rise" style={{ animationDelay: '200ms' }}>
          <p className="max-w-md text-lg leading-relaxed text-muted">
            A hybrid model reads the financial snapshot and the payment trajectory together, and
            returns a calibrated probability of default, with the reasons behind it.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-6">
            <a href="#cta" className="group inline-flex items-center gap-2 border-b border-ink pb-1 font-display text-base font-500 text-ink">
              Request access
              <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </a>
            <a href="#how" className="text-base text-muted transition-colors hover:text-ink">See how it works</a>
          </div>
        </div>
        <div className="rise" style={{ animationDelay: '320ms' }}>
          <LiveDemo />
        </div>
      </div>
      </div>
    </section>
  )
}

const WORDS = ['explainable', 'calibrated', 'auditable', 'deployable', 'fair', 'fast', 'sequence-aware']
function Marquee() {
  const row = [...WORDS, ...WORDS]
  return (
    <div className="mt-24 overflow-hidden border-y border-line py-5 md:mt-32">
      <div className="marquee flex w-max gap-10 whitespace-nowrap">
        {row.map((w, i) => (
          <span key={i} className="flex items-center gap-10 font-display text-2xl font-500 text-faint">
            {w} <span className="text-accent">/</span>
          </span>
        ))}
      </div>
    </div>
  )
}

const STATS = [
  { v: '0.89', l: 'model AUC on the hybrid' },
  { v: '<100ms', l: 'per decision, single call' },
  { v: '1000', l: 'applicants scored per batch' },
]
function Stats() {
  return (
    <section className="mx-auto max-w-[1240px] px-6 py-24 md:px-10 md:py-36">
      <div className="grid gap-12 md:grid-cols-3">
        {STATS.map((s) => (
          <div key={s.l} className="reveal border-t border-ink pt-6">
            <div className="font-display font-600 tracking-tight text-ink" style={{ fontSize: 'clamp(3rem,6vw,5rem)' }}>{s.v}</div>
            <div className="mt-3 max-w-[16rem] text-sm text-muted">{s.l}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

const STEPS = [
  { n: '01', t: 'Applicant data in', d: 'Financial facts and the recent monthly payment history, to one secure endpoint.' },
  { n: '02', t: 'Hybrid model scores', d: 'Gradient boosting reads the snapshot. A sequence model reads the payment trajectory. Their signals fuse.' },
  { n: '03', t: 'Decision out', d: 'A calibrated probability, an approve / review / decline call, and the factors behind it.' },
]
function How() {
  return (
    <section id="how" className="border-t border-line">
      <div className="mx-auto max-w-[1240px] px-6 py-24 md:px-10 md:py-36">
        <div className="reveal max-w-2xl">
          <Label>002 &nbsp;/&nbsp; how it works</Label>
          <h2 className="mt-6 font-display font-600 leading-[1.02] tracking-tight text-ink" style={{ fontSize: 'clamp(2rem,5vw,3.75rem)' }}>
            Two views of a borrower, combined.
          </h2>
        </div>
        <div className="mt-16 grid gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-3">
          {STEPS.map((s) => (
            <div key={s.n} className="reveal bg-bg p-8 md:p-10">
              <div className="font-display text-sm font-600 text-accent">{s.n}</div>
              <h3 className="mt-6 font-display text-xl font-600 text-ink">{s.t}</h3>
              <p className="mt-3 text-[15px] leading-relaxed text-muted">{s.d}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

const CAPS = [
  ['Explainable decisions', 'Every score ships with the factors that drove it. Defensible, contestable underwriting.'],
  ['Trajectory intelligence', 'The sequence model sees whether a borrower is improving or sliding, not just where they stand today.'],
  ['Deployable API', 'One authenticated endpoint. Into your origination stack in an afternoon.'],
  ['Fairness & audit trail', 'Every decision logged. Built to support bias audits and regulatory review.'],
  ['Batch portfolio scoring', 'One applicant or a thousand, in a single call.'],
  ['Calibrated probabilities', 'True probabilities, not just rankings. Price and reserve against them.'],
]
function Caps() {
  return (
    <section id="caps" className="border-t border-line">
      <div className="mx-auto max-w-[1240px] px-6 py-24 md:px-10 md:py-36">
        <div className="reveal"><Label>003 &nbsp;/&nbsp; capabilities</Label></div>
        <div className="mt-14 divide-y divide-line border-y border-line">
          {CAPS.map(([t, d], i) => (
            <div key={t} className="reveal group grid items-baseline gap-2 py-8 md:grid-cols-[6rem_1fr_1.4fr] md:gap-8">
              <span className="font-display text-sm text-faint">{String(i + 1).padStart(2, '0')}</span>
              <h3 className="font-display text-xl font-500 text-ink transition-colors group-hover:text-accent md:text-2xl">{t}</h3>
              <p className="text-[15px] leading-relaxed text-muted">{d}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function Output() {
  const factors = [['Recent payment history', '↑'], ['Debt-to-income ratio', '↑'], ['Employment history', '↓']]
  return (
    <section id="output" className="border-t border-line">
      <div className="mx-auto grid max-w-[1240px] gap-16 px-6 py-24 md:grid-cols-2 md:items-center md:px-10 md:py-36">
        <div className="reveal">
          <Label>004 &nbsp;/&nbsp; the output</Label>
          <h2 className="mt-6 font-display font-600 leading-[1.02] tracking-tight text-ink" style={{ fontSize: 'clamp(2rem,5vw,3.75rem)' }}>
            Not a black-box number.
          </h2>
          <p className="mt-6 max-w-md text-[15px] leading-relaxed text-muted">
            A decision report your officers can read and defend: the probability, the recommendation,
            and the factors that moved it, ready for an applicant or an auditor.
          </p>
        </div>
        <div className="reveal border border-ink bg-paper p-8 md:p-10">
          <div className="flex items-center justify-between border-b border-line pb-5">
            <span className="font-display text-sm text-muted">applicant #4821</span>
            <span className="font-display text-sm font-600 text-danger">decline</span>
          </div>
          <div className="py-8">
            <Label>probability of default</Label>
            <div className="mt-2 font-display font-700 tabular-nums text-danger" style={{ fontSize: 'clamp(3.5rem,7vw,5.5rem)', lineHeight: 1 }}>77.6<span className="text-3xl">%</span></div>
          </div>
          <div className="font-display text-sm font-600 text-ink">top factors</div>
          <ul className="mt-4 divide-y divide-line border-t border-line">
            {factors.map(([f, dir]) => (
              <li key={f} className="flex items-center justify-between py-3 text-[15px]">
                <span className="text-ink">{f}</span>
                <span className="font-display text-sm" style={{ color: dir === '↑' ? 'var(--color-danger)' : 'var(--color-success)' }}>{dir} risk</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}

function CTA() {
  return (
    <section id="cta" className="border-t border-line">
      <div className="mx-auto max-w-[1240px] px-6 py-28 text-center md:px-10 md:py-44">
        <h2 className="reveal mx-auto max-w-4xl font-display font-700 leading-[0.98] tracking-[-0.02em] text-ink" style={{ fontSize: 'clamp(2.5rem,7vw,6rem)' }}>
          Risk scoring you can <span className="italic font-500 text-accent">stand behind</span>.
        </h2>
        <div className="reveal mt-12 flex flex-wrap justify-center gap-8">
          <a href="#" className="group inline-flex items-center gap-2 border-b border-ink pb-1 font-display text-lg font-500 text-ink">
            Request access <ArrowUpRight className="h-5 w-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </a>
          <a href="#" className="text-lg text-muted transition-colors hover:text-ink">Talk to us</a>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1240px] flex-col items-start justify-between gap-4 px-6 py-10 text-sm text-muted md:flex-row md:items-center md:px-10">
        <span className="font-display text-ink">creditscore</span>
        <span>Demonstration project. Not validated for production lending.</span>
      </div>
    </footer>
  )
}

export default function App() {
  return (
    <>
      <ScrollProgress />
      <Nav />
      <main>
        <Hero />
        <Marquee />
        <ScrollStatement />
        <Stats />
        <How />
        <Caps />
        <Output />
        <CTA />
      </main>
      <Footer />
    </>
  )
}
