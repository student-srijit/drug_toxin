import { motion } from 'framer-motion'
import { CheckCircle2, XCircle } from 'lucide-react'

const METRIC_DEFS = [
  { key: 'LogP',  label: 'LogP',             desc: 'Lipophilicity', ideal: '< 5',  fmt: (v) => v?.toFixed(2) },
  { key: 'MW',    label: 'Mol. Weight',       desc: 'Daltons',       ideal: '< 500', fmt: (v) => v?.toFixed(1) },
  { key: 'QED',   label: 'QED Score',         desc: 'Drug-likeness', ideal: '> 0.5', fmt: (v) => v?.toFixed(3) },
  { key: 'TPSA',  label: 'TPSA',             desc: 'Å²',            ideal: '< 140', fmt: (v) => v?.toFixed(1) },
  { key: 'HBD',   label: 'H-Bond Donors',    desc: 'Count',         ideal: '≤ 5',  fmt: (v) => v },
  { key: 'HBA',   label: 'H-Bond Acceptors', desc: 'Count',         ideal: '≤ 10', fmt: (v) => v },
]

export default function PharmacologyCard({ profile }) {
  if (!profile) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="glass rounded-2xl border border-outline p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-display font-semibold text-sm text-white">Pharmacology Profile</h3>
          <p className="text-xs text-gray-500 mt-0.5">Drug-likeness & ADMET indicators</p>
        </div>
        <RO5Badge pass={profile.RO5 === 'PASS'} />
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        {METRIC_DEFS.map((m, i) => (
          <MetricTile key={m.key} def={m} value={profile[m.key]} delay={i * 0.06} />
        ))}
      </div>
    </motion.div>
  )
}

function MetricTile({ def, value, delay }) {
  const display = def.fmt(value)
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, delay }}
      className="bg-surface rounded-xl px-3 py-2.5 flex flex-col gap-0.5"
    >
      <span className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">{def.label}</span>
      <span className="font-display font-bold text-lg text-white mono leading-none">{display ?? '—'}</span>
      <span className="text-[10px] text-gray-600">{def.desc} · ideal {def.ideal}</span>
    </motion.div>
  )
}

function RO5Badge({ pass }) {
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border
      ${pass
        ? 'bg-green-400/10 border-green-400/30 text-green-400'
        : 'bg-red-400/10 border-red-400/30 text-red-400'
      }`}
    >
      {pass ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
      Lipinski {pass ? 'PASS' : 'FAIL'}
    </div>
  )
}
