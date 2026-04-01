import { motion } from 'framer-motion'
import { Box } from 'lucide-react'

export default function MolViewer3D({ smiles }) {
  if (!smiles) return null

  // We url-encode the SMILES string so it safely passes to our FastAPI /3d endpoint
  const viewerUrl = `/3d?smiles=${encodeURIComponent(smiles)}`

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      className="glass rounded-2xl border border-outline p-5 flex flex-col h-[340px]"
    >
      <div className="flex items-center gap-2 mb-3">
        <Box className="w-4 h-4 text-primary" />
        <h3 className="font-display font-semibold text-sm text-white">Interactive 3D Structure</h3>
      </div>
      
      <div className="flex-1 rounded-xl overflow-hidden border border-outline bg-[#070d1f] relative group">
        <iframe
          src={viewerUrl}
          className="w-full h-full border-none outline-none"
          title="3D Molecule Viewer"
        />
        <div className="absolute inset-0 pointer-events-none rounded-xl shadow-[inset_0_0_20px_rgba(0,0,0,0.6)]" />
        <div className="absolute bottom-2 right-3 text-[10px] text-gray-500 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
          Scroll to zoom · Drag to rotate
        </div>
      </div>
    </motion.div>
  )
}
