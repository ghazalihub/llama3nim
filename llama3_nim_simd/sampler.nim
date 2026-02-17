import std/[random, algorithm, math, sugar]
import tensor, model

proc newCategoricalSampler*(rng: ref Rand): Sampler =
  return proc(logits: FloatTensor): int =
    let r = rng[].rand(1.0f32)
    var cdf = 0.0f32
    for i in 0..<logits.size:
      cdf += logits.getFloat(i)
      if r < cdf: return i
    return logits.size - 1

proc siftDown(indices: var seq[int], logits: FloatTensor, start, n: int) =
  var prev = start
  while true:
    var next = 2 * prev + 1
    if next >= n: break
    let r = 2 * prev + 2
    if r < n and logits.getFloat(indices[r]) > logits.getFloat(indices[next]):
      next = r
    if logits.getFloat(indices[next]) > logits.getFloat(indices[prev]):
      let tmp = indices[prev]
      indices[prev] = indices[next]
      indices[next] = tmp
      prev = next
    else:
      break

proc newToppSampler*(vocabularySize: int, topp: float32, rng: ref Rand): Sampler =
  var indices = newSeq[int](vocabularySize)
  return proc(logits: FloatTensor): int =
    let n = logits.size
    var head = 0
    var tail = n - 1
    let cutoff = (1.0f32 - topp) / (n.float32 - 1.0f32)
    for i in 0..<n:
      if logits.getFloat(i) >= cutoff:
        indices[head] = i
        head += 1
      else:
        indices[tail] = i
        tail -= 1

    let n0 = head
    # Build max-heap (reversed comparator in Java)
    for i in countdown(n0 div 2 - 1, 0):
      siftDown(indices, logits, i, n0)

    var cumulativeProb = 0.0f32
    var lastIndex = 0
    for i in countdown(n0 - 1, 0):
      # swap 0 and i
      let tmp = indices[0]
      indices[0] = indices[i]
      indices[i] = tmp

      cumulativeProb += logits.getFloat(indices[i])
      if cumulativeProb > topp:
        lastIndex = i
        break
      siftDown(indices, logits, 0, i) # i is the new length

    let r = rng[].rand(1.0f32) * cumulativeProb
    var cdf = 0.0f32
    for i in countdown(n0 - 1, lastIndex):
      cdf += logits.getFloat(indices[i])
      if r < cdf: return indices[i]
    return indices[lastIndex]

proc selectSampler*(vocabularySize: int, temperature, topp: float32, rngSeed: int64): Sampler =
  if temperature == 0.0f32:
    return proc(logits: FloatTensor): int = logits.argmax()
  else:
    var rng = new(Rand)
    rng[] = initRand(rngSeed)
    let innerSampler = if topp <= 0 or topp >= 1:
      newCategoricalSampler(rng)
    else:
      newToppSampler(vocabularySize, topp, rng)

    return proc(logits: FloatTensor): int =
      logits.divideInPlace(0, logits.size, temperature)
      logits.softmaxInPlace(0, logits.size)
      return innerSampler(logits)
