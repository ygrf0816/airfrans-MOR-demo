import { useEffect, useState } from 'react'
import './App.css'
import Home from './pages/Home.tsx'
import Forward from './pages/Forward.tsx'
import Explain from './pages/Explain.tsx'
import Mor from './pages/Mor.tsx'
import Inverse from './pages/Inverse.tsx'
import Gallery from './pages/Gallery.tsx'
import SettingsPanel from './components/SettingsPanel.tsx'
import DataBadge from './components/DataBadge.tsx'
import { loadData, type DataState } from './data/load.ts'

export type PageKey = 'home' | 'mor' | 'forward' | 'explain' | 'inverse' | 'gallery'

const PAGE_KEYS: PageKey[] = ['home', 'mor', 'forward', 'explain', 'inverse', 'gallery']

const pageMeta: Record<PageKey, string> = {
  home: '主页',
  mor: '降维演示',
  forward: '前向预测',
  explain: '可解释性分析',
  inverse: '反向设计',
  gallery: '成果总览',
}

/** 从地址栏的 # 后面读页面名，认不出来就回主页。 */
function pageFromHash(): PageKey {
  const h = (typeof location === 'undefined' ? '' : location.hash.replace(/^#/, '')) as PageKey
  return PAGE_KEYS.includes(h) ? h : 'home'
}

function App() {
  const [active, setActive] = useState<PageKey>(pageFromHash)
  const [showSettings, setShowSettings] = useState(false)
  const [data, setData] = useState<DataState | null>(null)

  useEffect(() => {
    let alive = true
    loadData().then((s) => {
      if (alive) setData(s)
    })
    return () => {
      alive = false
    }
  }, [])

  // 页面与地址栏的 # 同步，浏览器前进后退也能用。
  useEffect(() => {
    const onHash = () => setActive(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const go = (key: PageKey) => {
    setActive(key)
    if (typeof location !== 'undefined' && location.hash !== '#' + key) {
      location.hash = key
    }
  }

  const renderPage = () => {
    if (!data) {
      return (
        <div className="page-in" style={{ display: 'grid', gap: 16 }}>
          <div className="skeleton" style={{ height: 120 }} />
          <div className="skeleton" style={{ height: 320 }} />
          <div className="skeleton" style={{ height: 160 }} />
        </div>
      )
    }
    if (active === 'home') return <Home data={data} go={go} />
    if (active === 'mor') return <Mor data={data} />
    if (active === 'forward') return <Forward data={data} />
    if (active === 'explain') return <Explain />
    if (active === 'inverse') return <Inverse data={data} />
    return <Gallery data={data} />
  }

  return (
    <div className="shell">
      <header className="shell-header">
        <div className="shell-title">
          <h1>翼型智能设计演示</h1>
          <span className="sub">AirfRANS 数据集 · 浏览器内推理</span>
          <DataBadge data={data} />
        </div>
        <nav className="shell-nav">
          {PAGE_KEYS.map((key) => (
            <button
              key={key}
              className={active === key ? 'nav-btn active' : 'nav-btn'}
              onClick={() => go(key)}
            >
              {pageMeta[key]}
            </button>
          ))}
          <button className="nav-btn" onClick={() => setShowSettings(true)}>
            设置
          </button>
        </nav>
      </header>
      {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}
      <main className="shell-main">{renderPage()}</main>
    </div>
  )
}

export default App
