import { motion, AnimatePresence } from 'framer-motion'
import { ShieldAlert, ShieldCheck } from 'lucide-react'

const SEV_CONFIG = {
  CRITICAL: {
    border: 'border-red-500/40',
    bg: 'bg-red-500/[0.07]',
    icon: 'text-red-400',
    badge: 'bg-red-500/20 text-red-400 border-red-400/30',
    dot: 'bg-red-400',
    emoji: '☠️',
  },
  HIGH: {
    border: 'border-orange-500/40',
    bg: 'bg-orange-500/[0.07]',
    icon: 'text-orange-400',
    badge: 'bg-orange-500/20 text-orange-400 border-orange-400/30',
    dot: 'bg-orange-400',
    emoji: '🚨',
  },
  MEDIUM: {
    border: 'border-yellow-500/40',
    bg: 'bg-yellow-500/[0.07]',
    icon: 'text-yellow-400',
    badge: 'bg-yellow-500/20 text-yellow-400 border-yellow-400/30',
    dot: 'bg-yellow-400',
    emoji: '⚠️',
  },
}

export default function AlertsPanel({ alerts }) {
  const hasAlerts = alerts?.length > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="glass rounded-2xl border border-outline p-5"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-display font-semibold text-sm text-white flex items-center gap-2">
            {hasAlerts
              ? <ShieldAlert className="w-4 h-4 text-red-400" />
              : <ShieldCheck className="w-4 h-4 text-green-400" />
            }
            Structural Safety Alerts
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Sub-structural toxic motif detection — beyond Tox21 assays
          </p>
        </div>
        {hasAlerts && (
          <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-400/30 shrink-0">
            {alerts.length} Alert{alerts.length > 1 ? 's' : ''}
          </span>
        )}
      </div>

      <AnimatePresence mode="wait">
        {hasAlerts ? (
          <div className="flex flex-col gap-2.5">
            {alerts.map((alert, i) => {
              const cfg = SEV_CONFIG[alert.severity] ?? SEV_CONFIG.MEDIUM
              return (
                <motion.div
                  key={alert.name}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className={`rounded-xl border p-4 ${cfg.bg} ${cfg.border}`}
                >
                  <div className="flex items-center gap-2.5 mb-1.5">
                    <span className="text-lg">{cfg.emoji}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="font-semibold text-sm text-white truncate">{alert.name}</p>
                        <span className={`text-[10px] font-bold uppercase tracking-wider border px-1.5 py-0.5 rounded-md ${cfg.badge}`}>
                          {alert.severity}
                        </span>
                      </div>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed">{alert.mechanism}</p>
                </motion.div>
              )
            })}

            {/* Disclaimer */}
            <div className="mt-1 p-3 rounded-xl bg-white/[0.03] border border-white/5">
              <p className="text-[10px] text-gray-500 leading-relaxed">
                ⚡ Structural alerts detect mechanisms that Tox21 in-vitro assays
                <em className="text-gray-400"> cannot</em> measure — including acetylcholinesterase inhibition,
                DNA alkylation, and heavy-metal chelation.
              </p>
            </div>
          </div>
        ) : (
          <motion.div
            key="clear"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center gap-2 py-8 text-center"
          >
            <div className="w-12 h-12 rounded-full bg-green-400/10 flex items-center justify-center">
              <ShieldCheck className="w-6 h-6 text-green-400" />
            </div>
            <p className="text-sm font-semibold text-green-400">No Structural Alerts Detected</p>
            <p className="text-xs text-gray-500 max-w-xs">
              This molecule does not match any known toxic sub-structural motifs in our alert library.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
