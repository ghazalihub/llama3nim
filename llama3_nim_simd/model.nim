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
  let config = model.config; result = State()
  result.x = newArrayFloatTensor(config.dim); result.xb = newArrayFloatTensor(config.dim)
  result.xb2 = newArrayFloatTensor(config.dim); result.hb = newArrayFloatTensor(config.hiddenDim)
  result.hb2 = newArrayFloatTensor(config.hiddenDim); result.q = newArrayFloatTensor(config.dim)
  result.k = newArrayFloatTensor(config.dim); result.v = newArrayFloatTensor(config.dim)
  result.att = newArrayFloatTensor(config.numberOfHeads * config.contextLength)
  result.logits = newArrayFloatTensor(config.vocabularySize)
  let kvDim = (config.dim * config.numberOfKeyValueHeads) div config.numberOfHeads
  result.keyCache = newSeq[FloatTensor](config.numberOfLayers)
  result.valueCache = newSeq[FloatTensor](config.numberOfLayers)
  for l in 0..<config.numberOfLayers:
    result.keyCache[l] = newArrayFloatTensor(config.contextLength * kvDim)
    result.valueCache[l] = newArrayFloatTensor(config.contextLength * kvDim)
  result.latestToken = model.tokenizer.specialTokens.getOrDefault("<|begin_of_text|>", 0)

proc attentionFusedWorker(pQ_base, pXb_base, pKeyCache_base, pValueCache_base: ptr float32, sH, eH: int, position, kvDim, headSize, kvMul: int, sqrtHeadSize: float32) =
  let num_regs = headSize div 8
  for h in sH..<eH:
    let qOffset = h * headSize; let xbOffset = h * headSize
    let pQa = cast[ptr UncheckedArray[float32]](cast[uint](pQ_base) + qOffset.uint * 4)
    let pXba = cast[ptr UncheckedArray[float32]](cast[uint](pXb_base) + xbOffset.uint * 4)
    let pKBa = cast[ptr UncheckedArray[float32]](cast[uint](pKeyCache_base) + (h div kvMul).uint * headSize.uint * 4)
    let pVBa = cast[ptr UncheckedArray[float32]](cast[uint](pValueCache_base) + (h div kvMul).uint * headSize.uint * 4)
    for i in 0..<headSize: pXba[i] = 0.0f32
    var m = -1e30f32; var s = 0.0f32
    for t in 0..position:
      let pK = cast[ptr UncheckedArray[float32]](addr pKBa[t * kvDim])
      var scorev = mm256_setzero_ps()
      for i in 0..<num_regs: scorev = mm256_fmadd_ps(mm256_load_ps(addr pQa[i * 8]), mm256_load_ps(addr pK[i * 8]), scorev)
      let score = hsum(scorev) / sqrtHeadSize
      let new_m = max(m, score); let fac_old = exp(m - new_m); let fac_new = exp(score - new_m)
      s = s * fac_old + fac_new; let vold = mm256_set1_ps(fac_old); let vnew = mm256_set1_ps(fac_new)
      let pV = cast[ptr UncheckedArray[float32]](addr pVBa[t * kvDim])
      for i in 0..<num_regs:
        let cur_v = mm256_load_ps(addr pXba[i * 8]); let new_v_t = mm256_load_ps(addr pV[i * 8])
        mm256_store_ps(addr pXba[i * 8], mm256_fmadd_ps(cur_v, vold, mm256_mul_ps(new_v_t, vnew)))
      m = new_m
    let inv_s = mm256_set1_ps(1.0f32 / s)
    for i in 0..<num_regs: mm256_store_ps(addr pXba[i * 8], mm256_mul_ps(mm256_load_ps(addr pXba[i * 8]), inv_s))

proc spawnMatmul(ctx: var Master, thiz, that, res: FloatTensor, dim0, dim1: int) =
  let numThreads = countProcessors(); let chunkSize = (dim0 + numThreads - 1) div numThreads
  let pA = cast[ptr float32](that.dataPtr); let pO = cast[ptr float32](res.dataPtr)
  case thiz.kind:
  of fkArray:
    let pW = cast[ptr float32](thiz.dataPtr)
    for t in 0..<numThreads:
      let s = t * chunkSize; let e = min(s + chunkSize, dim0)
      if s < e: ctx.spawn gemv_array_worker(pW, pA, pO, s, e, dim1)
  of fkQ8_0:
    let pW = cast[ptr byte](thiz.storage8)
    for t in 0..<numThreads:
      let s = t * chunkSize; let e = min(s + chunkSize, dim0)
      if s < e: ctx.spawn gemv_q8_worker(pW, pA, pO, s, e, dim1)
  of fkQ4_0:
    let pW = cast[ptr byte](thiz.storage4)
    for t in 0..<numThreads:
      let s = t * chunkSize; let e = min(s + chunkSize, dim0)
      if s < e: ctx.spawn gemv_q4_worker(pW, pA, pO, s, e, dim1)

proc forward*(ctx: var Master, model: LlamaModel, state: State, token: int, position: int): FloatTensor =
  let config = model.config; let weights = model.weights; let dim = config.dim; let headSize = config.headSize
  let kvDim = (config.dim * config.numberOfKeyValueHeads) div config.numberOfHeads
  let kvMul = config.numberOfHeads div config.numberOfKeyValueHeads; let sqrtHeadSize = sqrt(headSize.float32)
  weights.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim)
  for l in 0..<config.numberOfLayers:
    var t0 = cpuTime(); rmsnorm(state.xb, state.x, weights.rmsAttWeight[l], dim, config.rmsNormEps); globalProfile.rmsNormTime += cpuTime() - t0
    t0 = cpuTime()
    ctx.awaitAll:
      ctx.spawnMatmul(weights.wq[l], state.xb, state.q, dim, dim)
      ctx.spawnMatmul(weights.wk[l], state.xb, state.k, kvDim, dim)
      ctx.spawnMatmul(weights.wv[l], state.xb, state.v, kvDim, dim)
    globalProfile.matmulTime += cpuTime() - t0
    t0 = cpuTime()
    let pQa = state.q.dataPtr; let pKa = state.k.dataPtr; let pFcr = cast[ptr UncheckedArray[float32]](unsafeAddr weights.freqCisReal[0])
    let pFci = cast[ptr UncheckedArray[float32]](unsafeAddr weights.freqCisImag[0]); let sign_v = mm256_set_ps(1.0f32, -1.0f32, 1.0f32, -1.0f32, 1.0f32, -1.0f32, 1.0f32, -1.0f32)
    for i in countup(0, dim - 1, 8):
      let fcr8 = mm256_loadu_ps(addr pFcr[position * headSize + (i mod headSize)]); let fci8 = mm256_loadu_ps(addr pFci[position * headSize + (i mod headSize)])
      var qv = mm256_load_ps(addr pQa[i]); var qs = mm256_shuffle_ps(qv, qv, 0xB1); mm256_store_ps(addr pQa[i], mm256_fmadd_ps(mm256_mul_ps(sign_v, qs), fci8, mm256_mul_ps(qv, fcr8)))
      if i < kvDim: (var kv = mm256_load_ps(addr pKa[i]); var ks = mm256_shuffle_ps(kv, kv, 0xB1); mm256_store_ps(addr pKa[i], mm256_fmadd_ps(mm256_mul_ps(sign_v, ks), fci8, mm256_mul_ps(kv, fcr8))))
    globalProfile.ropeTime += cpuTime() - t0
    state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim); state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim)
    t0 = cpuTime(); let numThreads = countProcessors(); let headsPerThread = (config.numberOfHeads + numThreads - 1) div numThreads
    let pQ_base = cast[ptr float32](state.q.dataPtr); let pXb_base = cast[ptr float32](state.xb.dataPtr)
    let pKeyCache_base = cast[ptr float32](state.keyCache[l].dataPtr); let pValueCache_base = cast[ptr float32](state.valueCache[l].dataPtr)
    ctx.awaitAll:
      for t in 0..<numThreads:
        let sH = t * headsPerThread; let eH = min(sH + headsPerThread, config.numberOfHeads)
        if sH < eH: ctx.spawn attentionFusedWorker(pQ_base, pXb_base, pKeyCache_base, pValueCache_base, sH, eH, position, kvDim, headSize, kvMul, sqrtHeadSize)
    globalProfile.attnTime += cpuTime() - t0
    t0 = cpuTime(); ctx.matmulMT(weights.wo[l], state.xb, state.xb2, dim, dim); globalProfile.matmulTime += cpuTime() - t0
    state.x.addInPlace(state.xb2)
    t0 = cpuTime(); rmsnorm(state.xb, state.x, weights.rmsFfnWeight[l], dim, config.rmsNormEps); globalProfile.rmsNormTime += cpuTime() - t0
    t0 = cpuTime()
    ctx.awaitAll:
      ctx.spawnMatmul(weights.w1[l], state.xb, state.hb, config.hiddenDim, dim)
      ctx.spawnMatmul(weights.w3[l], state.xb, state.hb2, config.hiddenDim, dim)
    globalProfile.matmulTime += cpuTime() - t0
    siluMultiplyMT(ctx, state.hb, state.hb2)
    t0 = cpuTime(); ctx.matmulMT(weights.w2[l], state.hb, state.xb, dim, config.hiddenDim); globalProfile.matmulTime += cpuTime() - t0
    state.x.addInPlace(state.xb)
  var t0 = cpuTime(); rmsnorm(state.x, state.x, weights.rmsFinalWeight, dim, config.rmsNormEps); globalProfile.rmsNormTime += cpuTime() - t0
  t0 = cpuTime(); ctx.matmulMT(weights.wcls, state.x, state.logits, config.vocabularySize, dim); globalProfile.matmulTime += cpuTime() - t0
  return state.logits

proc generateTokens*(model: LlamaModel, state: State, startPosition: int, promptTokens: seq[int], stopTokens: HashSet[int], maxTokens: int, sampler: Sampler, echo: bool, onTokenGenerated: proc(t: int)): seq[int] =
  let startTime = cpuTime(); var maxToks = maxTokens
  if maxToks < 0 or model.config.contextLength < maxToks: maxToks = model.config.contextLength
  result = newSeq[int](); var token = state.latestToken; var nextToken: int; var promptIndex = 0
  if promptIndex < promptTokens.len and promptTokens[promptIndex] == token: promptIndex += 1
  var ctx = malebolgia.createMaster()
  for position in startPosition..<maxToks:
    let logits = forward(ctx, model, state, token, position)
    var t0 = cpuTime()
    if promptIndex < promptTokens.len:
      nextToken = promptTokens[promptIndex]; promptIndex += 1
      if echo: stderr.write(replaceControlCharacters(model.tokenizer.decode(@[nextToken])))
    else:
      nextToken = sampler(logits); if echo: stderr.write(replaceControlCharacters(model.tokenizer.decode(@[nextToken])))
      result.add(nextToken); if onTokenGenerated != nil: onTokenGenerated(nextToken)
      if stopTokens.contains(nextToken): break
    globalProfile.samplingTime += cpuTime() - t0
    token = nextToken; state.latestToken = token
  let elapsed = cpuTime() - startTime; globalProfile.totalTime = elapsed
  let totalTokens = promptIndex + result.len
  stderr.writeLine(&"\n{totalTokens.float / elapsed:.2f} tokens/s ({totalTokens})")
  stderr.writeLine("--- Profiling Results ---")
  stderr.writeLine(&"Matmul:  {globalProfile.matmulTime / elapsed * 100:5.2f}% ({globalProfile.matmulTime:5.2f}s)")
  stderr.writeLine(&"Attn:    {globalProfile.attnTime / elapsed * 100:5.2f}% ({globalProfile.attnTime:5.2f}s)")
  stderr.writeLine(&"RMSNorm: {globalProfile.rmsNormTime / elapsed * 100:5.2f}% ({globalProfile.rmsNormTime:5.2f}s)")
  stderr.writeLine(&"RoPE:    {globalProfile.ropeTime / elapsed * 100:5.2f}% ({globalProfile.ropeTime:5.2f}s)")
  stderr.writeLine(&"Sampling:{globalProfile.samplingTime / elapsed * 100:5.2f}% ({globalProfile.samplingTime:5.2f}s)")
  stderr.writeLine(&"Total:   100.00% ({elapsed:5.2f}s)")
