import { useEffect, useState } from 'react'

type Item = { id: string; param_name: string }

export default function ForecastViewer() {
  const [items, setItems] = useState<Item[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    fetch('/forecasts/manifest.json')
      .then((r) => r.json())
      .then((d) => {
        const list: Item[] = d?.items ?? []
        setItems(list)
        if (list.length > 0) setSelectedId(list[0].id)
      })
      .catch(() => setError('Kon manifest niet laden'))
      .finally(() => setLoading(false))
  }, [])

  const imgUrl = selectedId ? `/forecasts/${selectedId}.png` : ''

  return (
    <div className="w-full flex flex-col items-center gap-6">
      <div className="w-full max-w-xl">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Kies parameter
        </label>
        <select
          className="w-full border rounded-md px-3 py-2"
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          disabled={loading || !!error}
        >
          {items.map((it) => (
            <option key={it.id} value={it.id}>
              {it.id} — {it.param_name}
            </option>
          ))}
        </select>
        {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
      </div>

      {imgUrl ? (
        <div className="w-full max-w-4xl bg-white shadow rounded p-3">
          <img src={imgUrl} alt={selectedId} className="w-full h-auto" />
        </div>
      ) : (
        !loading && <p className="text-gray-500">Geen plot beschikbaar.</p>
      )}
    </div>
  )
}


