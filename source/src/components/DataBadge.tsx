import { useState } from 'react'
import { reloadData, type DataState } from '../data/load.ts'

/**
 * 数据来源角标：说明当前是示例数据还是真实数据。
 * 点击"重新检测"不刷新页面重新探测真实数据文件。
 */
export default function DataBadge({ data }: { data: DataState | null }) {
  const [busy, setBusy] = useState(false)
  if (!data) return null
  const isMock = data.mode === 'mock'
  return (
    <span style={{ display: 'inline-flex', gap: 8, alignItems: 'center' }}>
      <span className={isMock ? 'badge mock' : 'badge real'}>
        {isMock ? '示例数据' : '真实数据'}
      </span>
      {isMock && (
        <button
          style={{ fontSize: 11, padding: '2px 8px' }}
          disabled={busy}
          onClick={async () => {
            setBusy(true)
            await reloadData()
            // 数据状态通过顶层 effect 更新；这里简单刷新当前页面状态
            window.dispatchEvent(new Event('data-reloaded'))
            setBusy(false)
          }}
          title="重新探测 public/data 下的真实数据文件"
        >
          {busy ? '检测中' : '重新检测数据'}
        </button>
      )}
    </span>
  )
}
