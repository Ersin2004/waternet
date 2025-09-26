import './App.css'
import ForecastViewer from './ForecastViewer'

function App() {
  return (
    <div className="min-h-screen flex flex-col items-center p-6 gap-6 bg-gray-50">
      <h1 className="text-2xl font-bold">Forecast plots</h1>
      <ForecastViewer />
    </div>
  )
}

export default App
