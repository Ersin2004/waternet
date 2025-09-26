type Props = {
  url?: string
  height?: number
}

export default function StreamlitFrame({ url, height = 900 }: Props) {
  const src = url || (import.meta.env.VITE_STREAMLIT_URL as string) || 'http://localhost:8501'
  return (
    <div className="w-full flex justify-center">
      <iframe
        src={src}
        title="Streamlit Map"
        className="w-full max-w-6xl bg-white shadow rounded"
        style={{ height }}
      />
    </div>
  )
}


