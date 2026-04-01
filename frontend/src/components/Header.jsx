import { Activity, Dna, Zap } from 'lucide-react'

export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 glass border-b border-outline flex items-center px-6 gap-4">
      {/* Logo */}
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="w-8 h-8 rounded-xl bg-primary-gradient flex items-center justify-center shadow-lg shadow-primary/20">
          <Dna className="w-4 h-4 text-white" />
        </div>
        <span className="font-display font-700 text-sm text-white tracking-tight leading-none">
          Precision Drug Toxicity
          <span className="text-primary ml-1 font-semibold">Engine</span>
        </span>
      </div>

      {/* Center badge */}
      <div className="hidden md:flex items-center gap-1.5 ml-4 px-3 py-1 rounded-full bg-surface border border-outline">
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
        <span className="text-xs text-gray-400 font-medium">5-Model Ensemble · Live</span>
      </div>

      <div className="ml-auto flex items-center gap-3">
        {/* Quick stats */}
        <div className="hidden lg:flex items-center gap-4 mr-2">
          <Stat label="Models" value="5" icon={<Activity className="w-3 h-3" />} />
          <Stat label="Endpoints" value="12" icon={<Zap className="w-3 h-3" />} />
        </div>

        {/* Version */}
        <span className="text-xs font-mono text-gray-500 px-2 py-1 rounded-md bg-surface border border-outline">
          v2.0
        </span>
      </div>
    </header>
  )
}

function Stat({ label, value, icon }) {
  return (
    <div className="flex items-center gap-1.5 text-gray-400">
      <span className="text-primary">{icon}</span>
      <span className="text-xs font-semibold text-white">{value}</span>
      <span className="text-xs">{label}</span>
    </div>
  )
}
