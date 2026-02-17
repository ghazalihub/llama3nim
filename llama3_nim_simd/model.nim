import std/[math, times, strformat, sets, tables]
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

proc attnWorker(pQ, pKBase, pAtt, pXb, pVBase: ptr float32, position, kvDim, headSize, kvMul, contextLength: int, sqrtHeadSize: float32) =
  let pQa = cast[ptr UncheckedArray[float32]](pQ)
  let pKBa = cast[ptr UncheckedArray[float32]](pKBase)
  let pAtta = cast[ptr UncheckedArray[float32]](pAtt)
  let pXba = cast[ptr UncheckedArray[float32]](pXb)
  let pVBa = cast[ptr UncheckedArray[float32]](pVBase)

  for t in 0..position:
    let kOffset = t * kvDim
    let pK = addr pKBa[kOffset]
    var score = dotAvx(cast[ptr float32](pQa), cast[ptr float32](pK), headSize)
    score /= sqrtHeadSize
    pAtta[t] = score

  # Softmax
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
    let vOffset = t * kvDim
    let pV = addr pVBa[vOffset]
    let a = pAtta[t]
    let av = mm256_set1_ps(a)
    var i = 0
    let pVa = cast[ptr UncheckedArray[float32]](pV)
    while i <= headSize - 8:
      mm256_storeu_ps(addr pXba[i], mm256_fmadd_ps(av, mm256_loadu_ps(addr pVa[i]), mm256_loadu_ps(addr pXba[i])))
      i += 8
    while i < headSize:
      pXba[i] += a * pVa[i]
      i += 1

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

    # Vectorized RoPE
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
    ctx.awaitAll:
      for h in 0..<config.numberOfHeads:
        let pQh = addr state.q.data[h * headSize]
        let pKhBase = addr state.keyCache[l].data[(h div kvMul) * headSize]
        let pAtth = addr state.att.data[h * config.contextLength]
        let pXbh = addr state.xb.data[h * headSize]
        let pVhBase = addr state.valueCache[l].data[(h div kvMul) * headSize]
        ctx.spawn attnWorker(pQh, pKhBase, pAtth, pXbh, pVhBase, position, kvDim, headSize, kvMul, config.contextLength, sqrtHeadSize)
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

    let pHba = cast[ptr UncheckedArray[float32]](addr state.hb.data[0])
    let pHb2a = cast[ptr UncheckedArray[float32]](addr state.hb2.data[0])
    for i in 0..<config.hiddenDim:
      let v = pHba[i]
      pHba[i] = (v / (1.0f32 + exp(-v))) * pHb2a[i]

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

  # BOS fix as in scalar version
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
