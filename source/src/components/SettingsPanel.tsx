import { useEffect, useState } from 'react'
import {
  getSettings,
  isOnlineMode,
  setSettings,
  subscribeSettings,
  type LlmSettings,
} from '../lib/settings.ts'

interface SettingsPanelProps {
  onClose: () => void
}

// 密钥与接口配置只保存在内存变量中，禁止写入 localStorage、sessionStorage、
// IndexedDB、Cookie 或文件。关闭或刷新页面后全部丢弃。
export default function SettingsPanel({ onClose }: SettingsPanelProps) {
  const [form, setForm] = useState<LlmSettings>(getSettings())

  useEffect(() => subscribeSettings((s) => setForm(s)), [])

  const update = (patch: Partial<LlmSettings>) => {
    setSettings(patch)
  }

  const online = isOnlineMode(form)

  return (
    <>
      <div className="settings-backdrop" onClick={onClose} />
      <aside className="settings-panel">
        <div className="settings-head">
          <h3>大模型接口设置</h3>
          <button onClick={onClose}>关闭</button>
        </div>

        <p className="muted" style={{ margin: 0 }}>
          反向设计页面用大模型把一句话转成设计目标。不填接口时自动用本地规则模式，
          不联网也能完整演示。
        </p>

        <label>
          接口地址（OpenAI 兼容的 chat/completions 地址）
          <input
            type="text"
            placeholder="留空则使用本地规则模式"
            value={form.endpoint}
            onChange={(e) => update({ endpoint: e.target.value })}
          />
        </label>
        <label>
          模型名称
          <input
            type="text"
            placeholder="例如 gpt-4o-mini"
            value={form.model}
            onChange={(e) => update({ model: e.target.value })}
          />
        </label>
        <label>
          API key
          <input
            type="password"
            placeholder="只存在本页内存，关闭页面即丢失"
            value={form.apiKey}
            onChange={(e) => update({ apiKey: e.target.value })}
          />
        </label>
        <label className="row">
          <input
            type="checkbox"
            checked={form.localOnly}
            onChange={(e) => update({ localOnly: e.target.checked })}
          />
          仅本地规则模式（不需要任何密钥）
        </label>

        <p className="settings-note">
          密钥只保存在这个页面的内存里，不写本地存储，不上传仓库，关掉页面就没了。
        </p>
        <p className="settings-state">
          当前模式：
          <span className={online ? 'badge online' : 'badge mock'}>
            {online ? '在线模式' : '本地规则模式'}
          </span>
        </p>
      </aside>
    </>
  )
}
