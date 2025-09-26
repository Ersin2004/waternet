import { useEffect, useState } from 'react'

type Item = { file: string }

export default function ChemicalsViewer() {
  const [items, setItems] = useState<Item[]>([])
  const [selected, setSelected] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    // Try manifest; if not present, build a static list
    fetch('/graphs/manifest.json')
      .then((r) => r.json())
      .then((d) => {
        let raw = (d as any)?.items
        let list: Item[] = []
        if (Array.isArray(raw)) list = raw
        else if (raw && typeof raw === 'object') list = Object.values(raw)
        list = (list || []).filter((it: any) => typeof it?.file === 'string' && it.file.endsWith('.png'))
        if (!list.length) throw new Error('invalid manifest structure')
        setItems(list)
        setSelected(list[0].file)
      })
      .catch(() => {
        // fallback fixed list of known files
        const fallback: string[] = [
          'CHLFa_ug_l_spectro.png','E__m_L380nm.png','E__m_L440nm.png','FEOa_ug_l.png','GELDHD_mS_m_25oC.png',
          'MGETAL_mgHCO3_l_nf.png','NH3_mgN_l_nf.png','NH4_mgN_l_nf.png','NKj_mgN_l.png','NO2_mgN_l_nf.png',
          'NO3_mgN_l_nf.png','Ntot_mgN_l.png','O2__.png','O2_mg_l.png','PO4_mgP_l_nf.png','Ptot_mgP_l.png',
          'SO4_mg_l_nf.png','T_oC.png','ZICHT_m.png','ZS_mg_l.png','Cl_mg_l.png','Ca_mg_l_nf.png','pH.png'
        ]
        const list: Item[] = fallback.map(f => ({ file: f }))
        setItems(list)
        setSelected(list[0].file)
      })
      .finally(() => setLoading(false))
  }, [])

  const imgUrl = selected ? `/graphs/${selected}` : ''

  return (
    <div className="w-full flex flex-col items-center gap-4">
      <div className="w-full max-w-xl">
        <label className="block text-sm font-medium text-gray-700 mb-1">Kies chemical-plot</label>
        <select
          className="w-full border rounded-md px-3 py-2"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={loading || !!error}
        >
          {items.map((it) => (
            <option key={it.file} value={it.file}>{it.file.replace('.png','')}</option>
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


