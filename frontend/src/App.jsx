import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ShieldAlert, AlertCircle } from 'lucide-react'
import Header from './components/Header'
import HeroInput from './components/HeroInput'
import RiskBadge from './components/RiskBadge'
import PharmacologyCard from './components/PharmacologyCard'
import Tox21Table from './components/Tox21Table'
import AlertsPanel from './components/AlertsPanel'
import ShapChart from './components/ShapChart'
import MolViewer3D from './components/MolViewer3D'
import { predict } from './api'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [activeSmiles, setActiveSmiles] = useState('')

  const handlePredict = async (smiles) => {
    setActiveSmiles(smiles)
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await predict(smiles)
      if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'Failed to connect to the backend. Is the Python API running?'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen pb-24 text-gray-200">
      <Header />

      <main className="pt-24 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        {/* Intro */}
        <div className="text-center mb-10 max-w-2xl mx-auto mt-4 sm:mt-8">
          <motion.h1 
            initial={{ opacity: 0, y: -10 }} 
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl sm:text-4xl md:text-5xl font-display font-bold text-white mb-4 tracking-tight"
          >
            Predictive Molecular <span className="bg-clip-text text-transparent bg-primary-gradient">Toxicity.</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            transition={{ delay: 0.1 }}
            className="text-sm sm:text-base text-gray-400 mb-8"
          >
            Enter a SMILES notation to run our 5-model ensemble and structural motif scanner against 12 critical Tox21 endpoints.
          </motion.p>
          
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
            <HeroInput onSubmit={handlePredict} loading={loading} />
          </motion.div>
        </div>

        {/* Error handling */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="max-w-3xl mx-auto overflow-hidden"
            >
              <div className="glass border-red-500/30 bg-red-500/10 p-4 rounded-xl flex items-start gap-3 my-6">
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-red-400">Analysis Failed</h4>
                  <p className="text-sm text-red-200/70 mt-1">{error}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results grid */}
        <AnimatePresence mode="wait">
          {result && (
            <motion.div 
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-12"
            >
              <div className="flex items-center gap-3 mb-6">
                <ShieldAlert className="w-5 h-5 text-primary" />
                <h2 className="text-xl font-display font-semibold text-white">Analysis Report</h2>
                <div className="h-px bg-outline flex-1 ml-4" />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 auto-rows-max">
                {/* Left column - 1/3 */}
                <div className="flex flex-col gap-6 lg:col-span-1">
                  <RiskBadge 
                    overall={result.overall_risk} 
                    avgScore={result.average_score} 
                  />
                  <PharmacologyCard profile={result.pharmacology_profile} />
                  <AlertsPanel alerts={result.structural_alerts} />
                </div>

                {/* Right column - 2/3 */}
                <div className="flex flex-col gap-6 lg:col-span-2">
                  <Tox21Table predictions={result.predictions} />
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <MolViewer3D smiles={activeSmiles} />
                    <ShapChart smiles={activeSmiles} />
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  )
}
