import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BarChart2, ChevronDown, Loader2 } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer
} from 'recharts'
import { getShap } from '../api'

const TASKS = [
  'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER',
  'NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE',
  'SR-MMP','SR-p53',
]

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const val = payload[0].value
  return (
    <div className="glass px-3 py-2 rounded-lg border border-outline text-xs">
      <p className="text-white font-semibold truncate max-w-[180px]">{payload[0].payload.feature}</p>
      <p className={`font-mono font-bold mt-0.5 ${val >= 0 ? 'text-violet-400' : 'text-slate-400'}`}>
        SHAP: {val.toFixed(4)}
      </p>
    </div>
  )
}

export default function ShapChart({ smiles }) {
  const [task, setTask]       = useState('NR-AR')
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [open, setOpen]       = useState(false)
  const [error, setError]     = useState(null)

  useEffect(() => {
    if (!smiles) return
    setLoading(true)
    setError(null)
    getShap(smiles, task)
      .then((res) => {
        const sorted = [...(res.shap_top_features ?? [])].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        setData(sorted)
      })
      .catch(() => setError('SHAP analysis unavailable for this endpoint.'))
      .finally(() => setLoading(false))
  }, [smiles, task])

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.25 }}
      className="glass rounded-2xl border border-outline p-5"
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <h3 className="font-display font-semibold text-sm text-white flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-primary" />
            SHAP Interpretability
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">Top molecular features driving the prediction</p>
        </div>

        {/* Task selector */}
        <div className="relative">
          <button
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-surface border border-outline
                       text-xs font-medium text-gray-300 hover:border-primary/40 hover:text-white transition-all"
          >
            {task}
            <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />
          </button>
          <AnimatePresence>
            {open && (
              <motion.ul
                initial={{ opacity: 0, y: -4, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -4, scale: 0.97 }}
                transition={{ duration: 0.15 }}
                className="absolute right-0 mt-1 w-44 glass rounded-xl border border-outline py-1 z-20 shadow-2xl"
              >
                {TASKS.map((t) => (
                  <li key={t}>
                    <button
                      onClick={() => { setTask(t); setOpen(false) }}
                      className={`w-full text-left px-3 py-1.5 text-xs hover:bg-white/5 transition-colors
                        ${t === task ? 'text-primary font-semibold' : 'text-gray-300'}`}
                    >
                      {t}
                    </button>
                  </li>
                ))}
              </motion.ul>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Chart area */}
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div key="load" className="flex items-center justify-center h-56 gap-2 text-gray-500"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Loader2 className="w-4 h-4 animate-spin text-primary" />
            <span className="text-xs">Computing SHAP values…</span>
          </motion.div>
        ) : error ? (
          <motion.div key="err" className="flex items-center justify-center h-56"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <p className="text-xs text-gray-500">{error}</p>
          </motion.div>
        ) : data ? (
          <motion.div key={task} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} layout="vertical" margin={{ left: 80, right: 16, top: 4, bottom: 4 }}>
                <XAxis type="number" tick={{ fill: '#4b5563', fontSize: 10 }} axisLine={false} tickLine={false}
                  tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="feature" tick={{ fill: '#9ca3af', fontSize: 10 }}
                  axisLine={false} tickLine={false} width={78}
                  tickFormatter={(s) => s.length > 12 ? s.substring(0, 11) + '…' : s} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={12}>
                  {data.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.value >= 0
                        ? `rgba(129,140,248,${0.4 + 0.6 * (Math.abs(entry.value) / (data[0]?.value || 1))})`
                        : `rgba(100,116,139,${0.4 + 0.6 * (Math.abs(entry.value) / (Math.abs(data[0]?.value) || 1))})`
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <p className="mt-3 text-[10px] text-gray-600 leading-relaxed">
        Positive SHAP values (indigo) increase toxicity probability for this endpoint. Negative (slate) values reduce it.
      </p>
    </motion.div>
  )
}
