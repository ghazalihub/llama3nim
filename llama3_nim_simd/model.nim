import std/[math, times, strformat, sets, tables, cpuinfo]
import tensor, tokenizer, model_base, malebolgia, nimsimd/avx2, nimsimd/fma

type
  State* = ref object
    x*, xb*, xb2*, hb*, hb2*, q*, k*, v*, att*, logits*: FloatTensor
    keyCache*, valueCache*: seq[FloatTensor]
    latestToken*: int

  Sampler* = proc(logits: FloatTensor): int {.closure.}

  ProfileStats* = object
    matmulTime*, attnTime*, rmsNormTime*, softmaxTime*, samplingTime*, ropeTime*: float
    totalTime*: float

var globalProfile*: ProfileStats

proc newState*(model: LlamaModel): State =
  let config = model.config
  result = State()
  result.x = newArrayFloatTensor(config.dim)
  result.xb = newArrayFloatTensor(config.dim)
  result.xb2 = newArrayFloatTensor(config.dim)
  result.hb = newArrayFloatTensor(config.hiddenDim)
  result.hb2 = newArrayFloatTensor(config.hiddenDim)
  result.q = newArrayFloatTensor(config.dim)
  result.k = newArrayFloatTensor(config.dim)
  result.v = newArrayFloatTensor(config.dim)
  result.att = newArrayFloatTensor(config.numberOfHeads * config.contextLength)
  result.logits = newArrayFloatTensor(config.vocabularySize)

  let kvDim = (config.dim * config.numberOfKeyValueHeads) div config.numberOfHeads
  result.keyCache = newSeq[FloatTensor](config.numberOfLayers)
  result.valueCache = newSeq[FloatTensor](config.numberOfLayers)
  for l in 0..<config.numberOfLayers:
    result.keyCache[l] = newArrayFloatTensor(config.contextLength * kvDim)
    result.valueCache[l] = newArrayFloatTensor(config.contextLength * kvDim)

  result.latestToken = model.tokenizer.specialTokens.getOrDefault("<|begin_of_text|>", 0)

proc hsum(v: M256): float32 {.inline.} =
  let v128 = mm_add_ps(mm256_extractf128_ps(v, 1), mm256_castps256_ps128(v))
  let shuf = mm_movehdup_ps(v128)
  let sums = mm_add_ps(v128, shuf)
  let shuf2 = mm_movehl_ps(sums, sums)
  let sums2 = mm_add_ss(sums, shuf2)
  return mm_cvtss_f32(sums2)

proc multiHeadAttnWorker(pQ_base, pAtt_base, pXb_base, pKeyCache_base, pValueCache_base: ptr float32, sH, eH: int, position, kvDim, headSize, kvMul, contextLength: int, sqrtHeadSize: float32) =
  let num_regs = headSize div 8
  var q_regs: array[64, M256]

  for h in sH..<eH:
    let qOffset = h * headSize
    let attOffset = h * contextLength
    let xbOffset = h * headSize

    let pQa = cast[ptr UncheckedArray[float32]](cast[uint](pQ_base) + qOffset.uint * 4)
    let pKBa = cast[ptr UncheckedArray[float32]](cast[uint](pKeyCache_base) + (h div kvMul).uint * headSize.uint * 4)
    let pAtta = cast[ptr UncheckedArray[float32]](cast[uint](pAtt_base) + attOffset.uint * 4)
    let pXba = cast[ptr UncheckedArray[float32]](cast[uint](pXb_base) + xbOffset.uint * 4)
    let pVBa = cast[ptr UncheckedArray[float32]](cast[uint](pValueCache_base) + (h div kvMul).uint * headSize.uint * 4)

    for i in 0..<num_regs:
      q_regs[i] = mm256_loadu_ps(addr pQa[i * 8])

    for t in 0..position:
      let pK = cast[ptr UncheckedArray[float32]](addr pKBa[t * kvDim])
      var sumv = mm256_setzero_ps()
      for i in 0..<num_regs:
        sumv = mm256_fmadd_ps(q_regs[i], mm256_loadu_ps(addr pK[i * 8]), sumv)
      pAtta[t] = hsum(sumv) / sqrtHeadSize

    var maxVal = pAtta[0]
    for t in 1..position:
      if pAtta[t] > maxVal: maxVal = pAtta[t]
    var sum = 0.0f32
    for t in 0..position:
      pAtta[t] = exp(pAtta[t] - maxVal)
      sum += pAtta[t]
    for t in 0..position:
      pAtta[t] /= sum

    for i in 0..<headSize: pXba[i] = 0.0f32
    for t in 0..position:
      let a = pAtta[t]
      let av = mm256_set1_ps(a)
      let pV = cast[ptr UncheckedArray[float32]](addr pVBa[t * kvDim])
      for i in 0..<num_regs:
        mm256_storeu_ps(addr pXba[i * 8], mm256_fmadd_ps(av, mm256_loadu_ps(addr pV[i * 8]), mm256_loadu_ps(addr pXba[i * 8])))

proc forward*(ctx: var Master, model: LlamaModel, state: State, token: int, position: int): FloatTensor =
  let config = model.config
  let weights = model.weights
  let dim = config.dim
  let headSize = config.headSize
  let kvDim = (config.dim * config.numberOfKeyValueHeads) div config.numberOfHeads
  let kvMul = config.numberOfHeads div config.numberOfKeyValueHeads
  let sqrtHeadSize = sqrt(headSize.float32)

  weights.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim)

  for l in 0..<config.numberOfLayers:
    var t0 = cpuTime()
    rmsnorm(state.xb, state.x, weights.rmsAttWeight[l], dim, config.rmsNormEps)
    globalProfile.rmsNormTime += cpuTime() - t0

    t0 = cpuTime()
    ctx.matmulMT(weights.wq[l], state.xb, state.q, dim, dim)
    ctx.matmulMT(weights.wk[l], state.xb, state.k, kvDim, dim)
    ctx.matmulMT(weights.wv[l], state.xb, state.v, kvDim, dim)
    globalProfile.matmulTime += cpuTime() - t0

    t0 = cpuTime()
    let pQa = cast[ptr UncheckedArray[float32]](addr state.q.data[0])
    let pKa = cast[ptr UncheckedArray[float32]](addr state.k.data[0])
    let pFcr = cast[ptr UncheckedArray[float32]](unsafeAddr weights.freqCisReal[0])
    let pFci = cast[ptr UncheckedArray[float32]](unsafeAddr weights.freqCisImag[0])
    let sign_v = mm256_set_ps(1.0f32, -1.0f32, 1.0f32, -1.0f32, 1.0f32, -1.0f32, 1.0f32, -1.0f32)

    for i in countup(0, dim - 1, 8):
      let headDimBase = i mod headSize
      let fcr4 = mm_loadu_ps(addr pFcr[position * (headSize div 2) + (headDimBase div 2)])
      let fci4 = mm_loadu_ps(addr pFci[position * (headSize div 2) + (headDimBase div 2)])
      let fcr8 = mm256_set_m128(mm_unpackhi_ps(fcr4, fcr4), mm_unpacklo_ps(fcr4, fcr4))
      let fci8 = mm256_set_m128(mm_unpackhi_ps(fci4, fci4), mm_unpacklo_ps(fci4, fci4))

      var qv = mm256_loadu_ps(addr pQa[i])
      var qs = mm256_shuffle_ps(qv, qv, 0xB1)
      var resQ = mm256_fmadd_ps(mm256_mul_ps(sign_v, qs), fci8, mm256_mul_ps(qv, fcr8))
      mm256_storeu_ps(addr pQa[i], resQ)

      if i < kvDim:
        var kv = mm256_loadu_ps(addr pKa[i])
        var ks = mm256_shuffle_ps(kv, kv, 0xB1)
        var resK = mm256_fmadd_ps(mm256_mul_ps(sign_v, ks), fci8, mm256_mul_ps(kv, fcr8))
        mm256_storeu_ps(addr pKa[i], resK)
    globalProfile.ropeTime += cpuTime() - t0

    state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim)
    state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim)

    t0 = cpuTime()
    let numThreads = countProcessors()
    let headsPerThread = (config.numberOfHeads + numThreads - 1) div numThreads
    let pQ_base = addr state.q.data[0]
    let pAtt_base = addr state.att.data[0]
    let pXb_base = addr state.xb.data[0]
    let pKeyCache_base = addr state.keyCache[l].data[0]
    let pValueCache_base = addr state.valueCache[l].data[0]
    ctx.awaitAll:
      for t in 0..<numThreads:
        let sH = t * headsPerThread; let eH = min(sH + headsPerThread, config.numberOfHeads)
        if sH < eH: ctx.spawn multiHeadAttnWorker(pQ_base, pAtt_base, pXb_base, pKeyCache_base, pValueCache_base, sH, eH, position, kvDim, headSize, kvMul, config.contextLength, sqrtHeadSize)
    globalProfile.attnTime += cpuTime() - t0

    t0 = cpuTime()
    ctx.matmulMT(weights.wo[l], state.xb, state.xb2, dim, dim)
    globalProfile.matmulTime += cpuTime() - t0

    state.x.addInPlace(state.xb2)

    t0 = cpuTime()
    rmsnorm(state.xb, state.x, weights.rmsFfnWeight[l], dim, config.rmsNormEps)
    globalProfile.rmsNormTime += cpuTime() - t0

    t0 = cpuTime()
    ctx.matmulMT(weights.w1[l], state.xb, state.hb, config.hiddenDim, dim)
    ctx.matmulMT(weights.w3[l], state.xb, state.hb2, config.hiddenDim, dim)
    globalProfile.matmulTime += cpuTime() - t0

    siluMultiplyMT(ctx, state.hb, state.hb2)

    t0 = cpuTime()
    ctx.matmulMT(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim)
    globalProfile.matmulTime += cpuTime() - t0

    state.x.addInPlace(state.xb)

  var t0 = cpuTime()
  rmsnorm(state.x, state.x, weights.rmsFinalWeight, dim, config.rmsNormEps)
  globalProfile.rmsNormTime += cpuTime() - t0

  t0 = cpuTime()
  ctx.matmulMT(weights.wcls, state.x, state.logits, config.vocabularySize, dim)
  globalProfile.matmulTime += cpuTime() - t0

  return state.logits

proc generateTokens*(model: LlamaModel, state: State, startPosition: int, promptTokens: seq[int], stopTokens: HashSet[int], maxTokens: int, sampler: Sampler, echo: bool, onTokenGenerated: proc(t: int)): seq[int] =
  let startTime = cpuTime()
  var maxToks = maxTokens
  if maxToks < 0 or model.config.contextLength < maxToks:
    maxToks = model.config.contextLength

  result = newSeq[int]()
  var token = state.latestToken
  var nextToken: int
  var promptIndex = 0

  if promptIndex < promptTokens.len and promptTokens[promptIndex] == token:
    promptIndex += 1

  var ctx = malebolgia.createMaster()
  for position in startPosition..<maxToks:
    let logits = forward(ctx, model, state, token, position)

    var t0 = cpuTime()
    if promptIndex < promptTokens.len:
      nextToken = promptTokens[promptIndex]
      promptIndex += 1
      if echo:
        stderr.write(replaceControlCharacters(model.tokenizer.decode(@[nextToken])))
    else:
      nextToken = sampler(logits)
      if echo:
        stderr.write(replaceControlCharacters(model.tokenizer.decode(@[nextToken])))
      result.add(nextToken)
      if onTokenGenerated != nil:
        onTokenGenerated(nextToken)
      if stopTokens.contains(nextToken):
        break
    globalProfile.samplingTime += cpuTime() - t0

    token = nextToken
    state.latestToken = token

  let elapsed = cpuTime() - startTime
  globalProfile.totalTime = elapsed
  let totalTokens = promptIndex + result.len
  stderr.writeLine(&"\n{totalTokens.float / elapsed:.2f} tokens/s ({totalTokens})")

  stderr.writeLine("--- Profiling Results ---")
  stderr.writeLine(&"Matmul:  {globalProfile.matmulTime / elapsed * 100:5.2f}% ({globalProfile.matmulTime:5.2f}s)")
  stderr.writeLine(&"Attn:    {globalProfile.attnTime / elapsed * 100:5.2f}% ({globalProfile.attnTime:5.2f}s)")
  stderr.writeLine(&"RMSNorm: {globalProfile.rmsNormTime / elapsed * 100:5.2f}% ({globalProfile.rmsNormTime:5.2f}s)")
  stderr.writeLine(&"RoPE:    {globalProfile.ropeTime / elapsed * 100:5.2f}% ({globalProfile.ropeTime:5.2f}s)")
  stderr.writeLine(&"Sampling:{globalProfile.samplingTime / elapsed * 100:5.2f}% ({globalProfile.samplingTime:5.2f}s)")
  stderr.writeLine(&"Total:   100.00% ({elapsed:5.2f}s)")
