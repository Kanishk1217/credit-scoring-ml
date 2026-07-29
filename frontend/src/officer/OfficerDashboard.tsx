import { useState } from 'react'
import SingleView from './SingleView'
import BatchView from './BatchView'

type Tab = 'single' | 'batch'

function TopBar() {
  return (
    <header className="border-b border-line bg-bg">
      <div className="mx-auto flex max-w-[1240px] items-center justify-between px-6 py-4 md:px-10">
        <span className="font-display text-base font-600 tracking-tight text-ink">Agility</span>
        <div className="flex items-center gap-3 text-sm text-ink/55">
          <span>Loan officer</span>
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)]" title="API key configured" />
        </div>
      </div>
    </header>
  )
}

function TabBar({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  const Item = ({ id, label }: { id: Tab; label: string }) => (
    <button
      onClick={() => onChange(id)}
      className={`border-b-2 pb-3 font-display text-sm font-500 uppercase tracking-wide transition-colors ${tab === id ? 'border-accent text-ink' : 'border-transparent text-ink/45 hover:text-ink'}`}
    >
      {label}
    </button>
  )
  return (
    <div className="flex gap-8 border-b border-line">
      <Item id="single" label="Single applicant" />
      <Item id="batch" label="CSV batch" />
    </div>
  )
}

export default function OfficerDashboard() {
  const params = new URLSearchParams(window.location.search)
  const [tab, setTab] = useState<Tab>(params.get('tab') === 'batch' ? 'batch' : 'single')

  return (
    <div className="min-h-screen bg-bg">
      <TopBar />
      <main className="mx-auto max-w-[1240px] px-6 py-10 md:px-10">
        <TabBar tab={tab} onChange={setTab} />
        <div className="mt-8">
          {tab === 'single' ? <SingleView /> : <BatchView />}
        </div>
      </main>
    </div>
  )
}
