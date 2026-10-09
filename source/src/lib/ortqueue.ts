// ONNX Runtime 调用的全局串行队列。
//
// 为什么要这个。onnxruntime-web 的多线程 wasm 封装里有一个"当前会话"的全局槽，
// 代码大意是先记下本次会话句柄，跑完再核对句柄有没有被换掉，换掉了就抛
// "Session mismatch"。也就是说同一个 wasm 模块上不允许两次推理重叠执行。
//
// 前端有两处会并发调用推理。第一处是设计目标评估要跑四度、八度、十度三个攻角，
// 三个 Promise.all 一起发出去；第二处是压力曲线与流场云图分别用两个模块，
// 但底层是同一个 wasm 模块，也会重叠。实测表现就是会话建得好好的，
// 推理却总是报 Session mismatch 然后退回示例曲线。
//
// 办法很简单，所有推理调用都过这个队列，排队串行执行。
// 单次推理在毫秒级，串行带来的等待可以忽略。

let tail: Promise<unknown> = Promise.resolve()

/** 把一次推理排进队列，返回它的结果。前一个失败不影响后一个。 */
export function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
  const next = tail.then(fn, fn)
  tail = next.then(() => undefined, () => undefined)
  return next
}
