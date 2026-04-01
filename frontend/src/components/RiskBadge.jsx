import { motion } from 'framer-motion'

const RISK_CONFIG = {
  'CRITICAL':        { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',   border: 'rgba(239,68,68,0.35)',   icon: '☠️',  label: 'CRITICAL' },
  'DANGEROUS':       { color: '#f97316', bg: 'rgba(249,115,22,0.12)',  border: 'rgba(249,115,22,0.35)',  icon: '🚨', label: 'DANGEROUS' },
  'HIGH CONCERN':    { color: '#eab308', bg: 'rgba(234,179,8,0.12)',   border: 'rgba(234,179,8,0.35)',   icon: '⚠️', label: 'HIGH CONCERN' },
  'MODERATE CONCERN':{ color: '#f59e0b', bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.35)',  icon: '🔶', label: 'MODERATE CONCERN' },
  'LOW CONCERN':     { color: '#22c55e', bg: 'rgba(34,197,94,0.12)',   border: 'rgba(34,197,94,0.35)',   icon: '✅', label: 'LOW CONCERN' },
}

export default function RiskBadge({ overall, avgScore }) {
  const cfg = RISK_CONFIG[overall] ?? RISK_CONFIG['LOW CONCERN']

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
      style={{ background: cfg.bg, borderColor: cfg.border }}
      className="rounded-2xl border p-5 flex flex-col gap-3"
    >
      <p className="text-xs font-semibold uppercase tracking-widest text-gray-400">Overall Risk Assessment</p>

      <div className="flex items-center gap-3">
        <span className="text-3xl">{cfg.icon}</span>
        <div>
          <p className="font-display font-bold text-2xl leading-none" style={{ color: cfg.color }}>
            {cfg.label}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Avg. Tox21 Score:&nbsp;
            <span className="font-mono font-semibold text-white" style={{ color: cfg.color }}>
              {(avgScore * 100).toFixed(1)}%
            </span>
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(avgScore * 100, 100)}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${cfg.color}88, ${cfg.color})` }}
        />
      </div>
    </motion.div>
  )
}
