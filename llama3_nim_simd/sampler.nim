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
    var candidates = indices[0..<n0]
    candidates.sort((a, b) => cmp(logits.getFloat(b), logits.getFloat(a)))

    var cumulativeProb = 0.0f32
    var lastIndex = 0
    for i in 0..<n0:
      cumulativeProb += logits.getFloat(candidates[i])
      if cumulativeProb > topp:
        lastIndex = i
        break

    let r = rng[].rand(1.0f32) * cumulativeProb
    var cdf = 0.0f32
    for i in 0..lastIndex:
      cdf += logits.getFloat(candidates[i])
      if r < cdf: return candidates[i]
    return candidates[lastIndex]

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
