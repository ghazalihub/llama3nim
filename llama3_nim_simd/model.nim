import std/[math, sequtils, times, strformat, sets]
import tensor, tokenizer, model_base

type
  State* = ref object
    x*, xb*, xb2*, hb*, hb2*, q*, k*, v*, att*, logits*: FloatTensor
    keyCache*, valueCache*: seq[FloatTensor]
    latestToken*: int

  Sampler* = proc(logits: FloatTensor): int {.closure.}

  ProfileStats* = object
    matmulTime*, attnTime*, rmsNormTime*, softmaxTime*, samplingTime*: float
    totalTime*: float

var globalProfile*: ProfileStats

proc newState*(config: Configuration): State =
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

proc forward*(model: LlamaModel, state: State, token: int, position: int): FloatTensor =
  let config = model.config
  let weights = model.weights
  let dim = config.dim
  let headSize = config.headSize
  let kvDim = (config.dim * config.numberOfKeyValueHeads) div config.numberOfHeads
  let kvMul = config.numberOfHeads div config.numberOfKeyValueHeads
  let sqrtHeadSize = sqrt(headSize.float32)

  # copy the token embedding into x
  weights.tokenEmbeddingTable.copyTo(token * dim, state.x, 0, dim)

  for l in 0..<config.numberOfLayers:
    # attention rmsnorm
    var t0 = cpuTime()
    rmsnorm(state.xb, state.x, weights.rmsAttWeight[l], dim, config.rmsNormEps)
    globalProfile.rmsNormTime += cpuTime() - t0

    # qkv matmuls for this position
    t0 = cpuTime()
    weights.wq[l].matmul(state.xb, state.q, dim, dim)
    weights.wk[l].matmul(state.xb, state.k, kvDim, dim)
    weights.wv[l].matmul(state.xb, state.v, kvDim, dim)
    globalProfile.matmulTime += cpuTime() - t0

    # RoPE relative positional encoding
    for i in countup(0, dim - 1, 2):
      let headDim = i mod headSize
      let fcr = weights.freqCisReal[position * (headSize div 2) + (headDim div 2)]
      let fci = weights.freqCisImag[position * (headSize div 2) + (headDim div 2)]
      let rotn = if i < kvDim: 2 else: 1
      for v in 0..<rotn:
        let vec = if v == 0: state.q else: state.k
        let v0 = getFloat(vec, i)
        let v1 = getFloat(vec, i + 1)
        setFloat(vec, i, v0 * fcr - v1 * fci)
        setFloat(vec, i + 1, v0 * fci + v1 * fcr)

    # save key, value at this time step to our kv cache
    state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim)
    state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim)

    # multihead attention
    t0 = cpuTime()
    for h in 0..<config.numberOfHeads:
      let qOffset = h * headSize
      let attOffset = h * config.contextLength

      for t in 0..position:
        let keyCacheOffset = t * kvDim + (h div kvMul) * headSize
        var score = dot(state.q, state.keyCache[l], qOffset, keyCacheOffset, headSize)
        score /= sqrtHeadSize
        setFloat(state.att, attOffset + t, score)

      state.att.softmaxInPlace(attOffset, position + 1)

      let xbOffset = h * headSize
      state.xb.fillInPlace(xbOffset, headSize, 0.0f32)

      for t in 0..position:
        let vOffset = t * kvDim + (h div kvMul) * headSize
        let a = getFloat(state.att, attOffset + t)
        state.xb.saxpyInPlace(xbOffset, state.valueCache[l], vOffset, headSize, a)
    globalProfile.attnTime += cpuTime() - t0

    # final matmul to get the output of the attention
    t0 = cpuTime()
    weights.wo[l].matmul(state.xb, state.xb2, dim, dim)
    globalProfile.matmulTime += cpuTime() - t0

    # residual connection
    state.x.addInPlace(state.xb2)

    # ffn rmsnorm
    t0 = cpuTime()
    rmsnorm(state.xb, state.x, weights.rmsFfnWeight[l], dim, config.rmsNormEps)
    globalProfile.rmsNormTime += cpuTime() - t0

    # SwiGLU FFN
    t0 = cpuTime()
    weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim, dim)
    weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim, dim)
    globalProfile.matmulTime += cpuTime() - t0

    # SwiGLU non-linearity
    for i in 0..<config.hiddenDim:
      let v = getFloat(state.hb, i)
      setFloat(state.hb, i, v / (1.0f32 + exp(-v)))

    state.hb.multiplyInPlace(state.hb2)

    t0 = cpuTime()
    weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim)
    globalProfile.matmulTime += cpuTime() - t0

    # residual connection
    state.x.addInPlace(state.xb)

  # final rmsnorm
  var t0 = cpuTime()
  rmsnorm(state.x, state.x, weights.rmsFinalWeight, dim, config.rmsNormEps)
  globalProfile.rmsNormTime += cpuTime() - t0

  # classifier into logits
  t0 = cpuTime()
  weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim)
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

  for position in startPosition..<maxToks:
    let logits = forward(model, state, token, position)

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
  let totalTokens = (if promptIndex > 0: promptIndex else: 0) + result.len
  stderr.writeLine(&"\n{totalTokens.float / elapsed:.2f} tokens/s ({totalTokens})")

  # Print profiling results
  stderr.writeLine("--- Profiling Results ---")
  stderr.writeLine(&"Matmul:  {globalProfile.matmulTime / elapsed * 100:5.2f}% ({globalProfile.matmulTime:5.2f}s)")
  stderr.writeLine(&"Attn:    {globalProfile.attnTime / elapsed * 100:5.2f}% ({globalProfile.attnTime:5.2f}s)")
  stderr.writeLine(&"RMSNorm: {globalProfile.rmsNormTime / elapsed * 100:5.2f}% ({globalProfile.rmsNormTime:5.2f}s)")
  stderr.writeLine(&"Sampling:{globalProfile.samplingTime / elapsed * 100:5.2f}% ({globalProfile.samplingTime:5.2f}s)")
  stderr.writeLine(&"Total:   100.00% ({elapsed:5.2f}s)")
