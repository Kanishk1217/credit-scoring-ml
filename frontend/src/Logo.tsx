/* The mark: a solid disc cut flat on one edge, with a dot on the cut -- a score capped at a
   threshold. Boxed in the ink tile so it also works standalone as the favicon (see
   public/favicon.svg, which mirrors this path data). */
export default function Logo({ className = '' }: { className?: string }) {
  return (
    <a href="/" className={`inline-flex items-center gap-2.5 ${className}`}>
      <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true" className="shrink-0">
        <rect width="32" height="32" rx="7" className="fill-ink" />
        <g transform="translate(3,3)">
          <path d="M13 3a10 10 0 1 0 5 18.66L18 13Z" className="fill-accent" />
          <circle cx="18" cy="13" r="1.8" className="fill-paper" />
        </g>
      </svg>
      <span className="font-display text-base font-600 tracking-tight text-ink">creditscore</span>
    </a>
  )
}
