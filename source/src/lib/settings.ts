// 全局设置：接口地址、模型名称、API key 只保存在内存里。
// 不写 localStorage、sessionStorage、IndexedDB、Cookie 或文件。
// 页面关闭或刷新后全部丢失。代码里不出现任何默认密钥。

export interface LlmSettings {
  endpoint: string
  model: string
  apiKey: string
  localOnly: boolean
}

let current: LlmSettings = {
  endpoint: '',
  model: '',
  apiKey: '',
  localOnly: false,
}

const subs = new Set<(s: LlmSettings) => void>()

export function getSettings(): LlmSettings {
  return { ...current }
}

export function setSettings(next: Partial<LlmSettings>): void {
  current = { ...current, ...next }
  subs.forEach((fn) => fn(getSettings()))
}

export function subscribeSettings(fn: (s: LlmSettings) => void): () => void {
  subs.add(fn)
  return () => subs.delete(fn)
}

export function isOnlineMode(s: LlmSettings): boolean {
  return !s.localOnly && s.endpoint.trim() !== '' && s.apiKey.trim() !== ''
}
