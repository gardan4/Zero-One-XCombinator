import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

// NOTE: StrictMode intentionally omitted. In dev it double-invokes effects, which fires every model
// prediction twice — the Live-compare page would then hit DeepSeek/Featherless with two concurrent
// requests and trip its free-tier 1-request cap. One render = one request keeps the live demo clean.
createRoot(document.getElementById('root')!).render(<App />)
