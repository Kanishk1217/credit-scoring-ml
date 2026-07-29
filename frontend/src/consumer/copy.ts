// All user-facing strings live here so the tone stays consistent: never "denied", "rejected",
// "high risk", "bad" -- always forward-looking ("readiness", "building", "your plan", "unlocks").

export const COPY = {
  heading: 'Where do you stand?',
  headingAccent: 'you',
  sub: "A quick, private read on where you stand today, and what would move you forward.",
  submit: 'See where I stand',
  submitting: 'Reading your details…',
  startOver: 'Start over',
  editInputs: 'Edit my details',
  csvHint: 'Or upload a one-row CSV with your details instead.',
  errorFallback: "That didn't go through. Your details are safe — try again.",
  readinessLabel: 'Readiness',
  whyLabel: "What's shaping this",
  adviceLabel: 'Your plan forward',
  goalLabel: 'The path to qualifying',
  goalReached: "You're already over the line — nice work.",
  goalReachable: (n: number) => `${n} move${n === 1 ? '' : 's'} put${n === 1 ? 's' : ''} you over the line.`,
  goalUnreachable: "These moves get you closer. Here's the best offer on the table today.",
  offerNowLabel: 'Today',
  offerAfterLabel: 'After your plan',
}

export const EFFORT_LABEL: Record<string, string> = {
  time: 'Takes time',
  money: 'Costs money',
  habit: 'A habit to build',
}
