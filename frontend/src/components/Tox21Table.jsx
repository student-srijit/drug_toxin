import { motion } from 'framer-motion'

const RISK_STYLE = {
  HIGH:   { bar: '#ef4444', text: 'text-red-400',    bg: 'bg-red-400/10' },
  MEDIUM: { bar: '#eab308', text: 'text-yellow-400', bg: 'bg-yellow-400/10' },
  LOW:    { bar: '#22c55e', text: 'text-green-400',  bg: 'bg-green-400/10' },
}

const PATHWAY = {
  'NR-AR':         'Androgen Receptor',
  'NR-AR-LBD':     'AR — Ligand Binding',
  'NR-AhR':        'Aryl Hydrocarbon R.',
  'NR-Aromatase':  'Aromatase',
  'NR-ER':         'Estrogen Receptor',
  'NR-ER-LBD':     'ER — Ligand Binding',
  'NR-PPAR-gamma': 'PPAR-γ',
  'SR-ARE':        'Antioxidant Response',
  'SR-ATAD5':      'Genotoxicity / DNA',
  'SR-HSE':        'Heat Shock Response',
  'SR-MMP':        'Mitochondrial Memb.',
  'SR-p53':        'p53 Tumor Suppressor',
}

export default function Tox21Table({ predictions }) {
  if (!predictions?.length) return null
  const sorted = [...predictions].sort((a, b) => b.Ensemble - a.Ensemble)

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.15 }}
      className="glass rounded-2xl border border-outline overflow-hidden"
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-3 border-b border-outline">
        <h3 className="font-display font-semibold text-sm text-white">Tox21 Endpoint Predictions</h3>
        <p className="text-xs text-gray-500 mt-0.5">12 nuclear receptor & stress-response assays</p>
      </div>

      {/* Column labels */}
      <div className="grid grid-cols-[1fr_auto_auto_80px] gap-x-4 px-5 py-2 text-[10px] uppercase tracking-wider font-semibold text-gray-600 border-b border-outline">
        <span>Endpoint / Pathway</span>
        <span className="text-right">Score</span>
        <span className="text-right">Risk</span>
        <span>Probability</span>
      </div>

      {/* Rows */}
      <div className="divide-y divide-outline">
        {sorted.map((row, i) => {
          const cfg = RISK_STYLE[row.Risk] ?? RISK_STYLE.LOW
          const pct = Math.min(row.Ensemble * 100, 100)
          return (
            <motion.div
              key={row.Task}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
              className="tox-row grid grid-cols-[1fr_auto_auto_80px] gap-x-4 items-center px-5 py-3"
            >
              {/* Task name */}
              <div>
                <p className="text-sm font-medium text-white">{row.Task}</p>
                <p className="text-[10px] text-gray-500">{PATHWAY[row.Task] ?? ''}</p>
              </div>

              {/* Score */}
              <span className={`mono text-sm font-semibold ${cfg.text}`}>
                {(row.Ensemble * 100).toFixed(1)}%
              </span>

              {/* Risk badge */}
              <span className={`text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full ${cfg.bg} ${cfg.text}`}>
                {row.Risk}
              </span>

              {/* Probability bar */}
              <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.6, delay: 0.1 + i * 0.04, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{ background: cfg.bar }}
                />
              </div>
            </motion.div>
          )
        })}
      </div>
    </motion.div>
  )
}
