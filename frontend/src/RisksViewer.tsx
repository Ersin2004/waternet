import { useEffect, useState } from 'react'

type RiskItem = { file: string }

export default function RisksViewer() {
  const [items, setItems] = useState<RiskItem[]>([])
  const [selected, setSelected] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    // Try manifest; if missing, fallback to known files
    fetch('/risks/manifest.json')
      .then((r) => r.json())
      .then((d) => {
        let raw = (d as any)?.items
        let list: RiskItem[] = []
        if (Array.isArray(raw)) {
          list = raw
        } else if (raw && typeof raw === 'object') {
          try {
            list = Object.values(raw)
          } catch {
            list = []
          }
        }
        list = (list || []).filter((it: any) => typeof it?.file === 'string' && it.file.length > 0)
        if (!list.length) {
          throw new Error('Invalid manifest structure')
        }
        setItems(list)
        setSelected(list[0].file)
      })
      .catch(() => {
        const fallback: RiskItem[] = [
          { file: 'DANGER_DASHBOARD.png' },
          { file: 'DANGEROUS_MOMENTS_TIMELINE.png' },
          { file: 'NH3_TIME.png' },
          { file: 'PH_AMMONIUM_DANGER.png' },
        ]
        setItems(fallback)
        setSelected(fallback[0].file)
      })
      .finally(() => setLoading(false))
  }, [])

  const imgUrl = selected ? `/risks/${selected}` : ''

  return (
    <div className="w-full flex flex-col items-center gap-4">
      <div className="w-full max-w-xl">
        <label className="block text-sm font-medium text-gray-700 mb-1">Kies risk-plot</label>
        <select
          className="w-full border rounded-md px-3 py-2"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={loading || !!error}
        >
          {items.map((it) => (
            <option key={it.file} value={it.file}>
              {it.file}
            </option>
          ))}
        </select>
        {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
      </div>

      {imgUrl ? (
        <div className="w-full max-w-5xl bg-white shadow rounded p-3">
          <img src={imgUrl} alt={selected} className="w-full h-auto" />
        </div>
      ) : (
        !loading && <p className="text-gray-500">Geen afbeelding beschikbaar.</p>
      )}
    </div>
  )
}


