import './App.css'
import ForecastViewer from './ForecastViewer'
import CorrelationHeatmap from './CorrelationHeatmap'
import { useState } from 'react'
import RisksViewer from './RisksViewer'
import ChemicalsViewer from './ChemicalsViewer'
import StreamlitFrame from './StreamlitFrame'

function App() {
  const [view, setView] = useState<'forecast' | 'heatmap' | 'risks' | 'chemicals' | 'map'>('forecast')

  return (
    <div className="min-h-screen flex flex-col items-center p-6 gap-6 bg-gray-50">
      <h1 className="text-2xl font-bold">Waternet dashboard</h1>

      <div className="w-full max-w-xl">
        <label className="block text-sm font-medium text-gray-700 mb-1">Kies weergave</label>
        <select
          className="w-full border rounded-md px-3 py-2"
          value={view}
          onChange={(e) => setView(e.target.value as any)}
        >
          <option value="forecast">Forecast</option>
          <option value="heatmap">Heatmap</option>
          <option value="risks">Risks</option>
          <option value="chemicals">Chemicals</option>
          <option value="map">Map</option>
        </select>
      </div>

      {view === 'forecast' && <ForecastViewer />}
      {view === 'heatmap' && <CorrelationHeatmap />}
      {view === 'risks' && <RisksViewer />}
      {view === 'chemicals' && <ChemicalsViewer />}
      {view === 'map' && <StreamlitFrame />}
    </div>
  )
}

export default App
