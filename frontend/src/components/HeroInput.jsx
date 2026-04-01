import { useState } from 'react'
import { Search, Sparkles, ChevronDown, ArrowRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

const EXAMPLES = [
  { name: 'Aspirin',        smiles: 'CC(=O)Oc1ccccc1C(=O)O',          tag: 'Safe',     color: 'text-green-400' },
  { name: 'Caffeine',       smiles: 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',      tag: 'Safe',     color: 'text-green-400' },
  { name: 'Ibuprofen',      smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',       tag: 'Moderate', color: 'text-yellow-400' },
  { name: 'Paracetamol',    smiles: 'CC(=O)Nc1ccc(O)cc1',               tag: 'Low',      color: 'text-green-400' },
  { name: 'Sarin',          smiles: 'COP(=O)(OC(C)C)F',                  tag: 'CRITICAL', color: 'text-red-400' },
  { name: 'Mustard Gas',    smiles: 'ClCCSCCCl',                          tag: 'CRITICAL', color: 'text-red-400' },
]

export default function HeroInput({ onSubmit, loading }) {
  const [smiles, setSmiles] = useState('')
  const [open, setOpen]     = useState(false)

  const submit = (s) => {
    const val = (s ?? smiles).trim()
    if (!val || loading) return
    setSmiles(val)
    onSubmit(val)
    setOpen(false)
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Main input row */}
      <div
        className="flex items-center gap-3 glass rounded-2xl px-4 py-3 border border-outline
                   focus-within:border-primary/50 focus-within:shadow-[0_0_0_3px_rgba(129,140,248,0.1)]
                   transition-all duration-200"
      >
        <Search className="w-5 h-5 text-gray-500 shrink-0" />

        <input
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          placeholder="Enter SMILES notation  e.g. CC(=O)Oc1ccccc1C(=O)O"
          className="flex-1 bg-transparent text-sm text-white font-mono placeholder:text-gray-600
                     placeholder:font-sans focus:outline-none"
          spellCheck={false}
        />

        {/* Example dropdown trigger */}
        <button
          onClick={() => setOpen((o) => !o)}
          className="hidden sm:flex items-center gap-1 text-xs text-gray-400 hover:text-primary
                     transition-colors px-2 py-1 rounded-lg hover:bg-primary/10 shrink-0"
        >
          <Sparkles className="w-3.5 h-3.5" />
          Examples
          <ChevronDown className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>

        {/* Predict button */}
        <button
          onClick={() => submit()}
          disabled={!smiles.trim() || loading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold text-white
                     bg-primary-gradient hover:opacity-90 active:scale-95
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all duration-150 shadow-lg shadow-primary/20 shrink-0"
        >
          {loading ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analyzing
            </>
          ) : (
            <>
              Predict
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>

      {/* Example dropdown */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.18 }}
            className="mt-2 glass rounded-2xl border border-outline overflow-hidden"
          >
            <div className="p-2 grid grid-cols-2 sm:grid-cols-3 gap-1">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.name}
                  onClick={() => {
                    setSmiles(ex.smiles)
                    submit(ex.smiles)
                  }}
                  className="flex items-center justify-between px-3 py-2.5 rounded-xl
                             hover:bg-white/5 transition-colors text-left group"
                >
                  <div>
                    <p className="text-sm font-medium text-white group-hover:text-primary transition-colors">
                      {ex.name}
                    </p>
                    <p className="text-[10px] font-mono text-gray-600 truncate max-w-[120px] mt-0.5">
                      {ex.smiles.substring(0, 18)}…
                    </p>
                  </div>
                  <span className={`text-[10px] font-semibold uppercase tracking-wider ${ex.color}`}>
                    {ex.tag}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
