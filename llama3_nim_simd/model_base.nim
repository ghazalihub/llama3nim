import std/[streams, tables, sequtils, math, memfiles, strutils, os]
import gguf, tensor, tokenizer

type
  Configuration* = object
    dim*, hiddenDim*, numberOfLayers*, numberOfHeads*, numberOfKeyValueHeads*: int
    vocabularySize*, contextLength*: int
    rmsNormEps*, ropeTheta*: float32
    headSize*: int

  Weights* = ref object
    tokenEmbeddingTable*: FloatTensor
    rmsAttWeight*: seq[seq[float32]]
    wq*, wk*, wv*, wo*: seq[FloatTensor]
    rmsFfnWeight*: seq[seq[float32]]
    w1*, w2*, w3*: seq[FloatTensor]
    rmsFinalWeight*: seq[float32]
    freqCisReal*, freqCisImag*: seq[float32]
    wcls*: FloatTensor

  LlamaModel* = ref object
    config*: Configuration
    tokenizer*: Tokenizer
    weights*: Weights
    memFile*: MemFile

proc precomputeFreqsCis*(contextLength, headSize: int, theta: float64,
                         ropeScaling: bool, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength: float32): (seq[float32], seq[float32]) =
  var cr = newSeq[float32](contextLength * (headSize div 2))
  var ci = newSeq[float32](contextLength * (headSize div 2))
  var n = 0
  for pos in 0..<contextLength:
    for i in countup(0, headSize - 1, 2):
      var freq = 1.0f32 / pow(theta.float32, (i.float32 / headSize.float32))
      if ropeScaling:
        let loFreqWavelen = oldContextLength / loFreqFactor
        let hiFreqWavelen = oldContextLength / hiFreqFactor
        let wavelen = 2.0f32 * PI.float32 / freq
        if wavelen < hiFreqWavelen:
          discard
        elif wavelen > loFreqWavelen:
          freq = freq / scaleFactor
        else:
          let smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor)
          freq = (1.0f32 - smooth) * freq / scaleFactor + smooth * freq

      let val = pos.float32 * freq
      cr[n] = cos(val)
      ci[n] = sin(val)
      n += 1
  return (cr, ci)

proc loadQuantized(ti: GGUFTensorInfo, m: MemFile, dataOffset: int64): FloatTensor =
  let size = ti.dimensions.foldl(a * b, 1'i64).int
  let p = cast[ptr UncheckedArray[byte]](cast[int](m.mem) + dataOffset.int + ti.offset.int)
  case ti.ggmlType:
  of Q4_0: result = FloatTensor(kind: fkQ4_0, size: size, storage4: p)
  of Q8_0: result = FloatTensor(kind: fkQ8_0, size: size, storage8: p)
  of F32:
    let data = cast[ptr UncheckedArray[float32]](p)
    var f32data = newSeq[float32](size)
    for i in 0..<size: f32data[i] = data[i]
    result = FloatTensor(kind: fkArray, size: size, data: f32data)
  else: raise newException(ValueError, "Unsupported GGML type: " & $ti.ggmlType)

proc toFloatSeq(ti: GGUFTensorInfo, m: MemFile, dataOffset: int64): seq[float32] =
  let size = ti.dimensions.foldl(a * b, 1'i64).int
  let p = cast[ptr UncheckedArray[float32]](cast[int](m.mem) + dataOffset.int + ti.offset.int)
  result = newSeq[float32](size)
  for i in 0..<size: result[i] = p[i]

proc getInt(v: MetadataValue): int =
  case v.kind:
  of mvtUint32: result = v.u32.int
  of mvtInt32: result = v.i32.int
  of mvtUint64: result = v.u64.int
  of mvtInt64: result = v.i64.int
  of mvtUint16: result = v.u16.int
  of mvtInt16: result = v.i16.int
  of mvtUint8: result = v.u8.int
  of mvtInt8: result = v.i8.int
  else: raise newException(ValueError, "Not an integer kind: " & $v.kind)

proc loadModel*(filename: string, contextLength: int): LlamaModel =
  let gguf = loadGGUF(filename)
  let m = memfiles.open(filename)

  # madvise for performance on Linux
  when defined(linux):
    proc madvise(addrp: pointer, length: int, advice: int): int {.header: "<sys/mman.h>", importc: "madvise".}
    const MADV_RANDOM = 1
    const MADV_WILLNEED = 3
    discard madvise(m.mem, m.size, MADV_WILLNEED)

  let metadata = gguf.metadata
  let vocabTokens = metadata["tokenizer.ggml.tokens"].arr.mapIt(it.s)
  let vocabulary = newVocabulary(vocabTokens)

  var merges = newSeq[(int, int)]()
  if metadata.contains("tokenizer.ggml.merges"):
    for line in metadata["tokenizer.ggml.merges"].arr:
      let parts = line.s.split(' ')
      if parts.len == 2:
        let f = vocabulary.tokenToIndex.getOrDefault(parts[0], -1)
        let s = vocabulary.tokenToIndex.getOrDefault(parts[1], -1)
        if f != -1 and s != -1:
          merges.add((f, s))

  var specialTokens = initTable[string, int]()
  let baseTokens = 128000
  if vocabulary.tokens.len > baseTokens:
    for i in baseTokens..<vocabulary.tokens.len:
      specialTokens[vocabulary.tokens[i]] = i

  let tokenizer = newTokenizer(vocabulary, merges, "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", specialTokens)

  var config = Configuration(
    dim: metadata["llama.embedding_length"].getInt(),
    hiddenDim: metadata["llama.feed_forward_length"].getInt(),
    numberOfLayers: metadata["llama.block_count"].getInt(),
    numberOfHeads: metadata["llama.attention.head_count"].getInt(),
    vocabularySize: vocabulary.tokens.len,
    contextLength: metadata["llama.context_length"].getInt(),
    rmsNormEps: metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", MetadataValue(kind: mvtFloat32, f32: 1e-5f)).f32,
    ropeTheta: metadata.getOrDefault("llama.rope.freq_base", MetadataValue(kind: mvtFloat32, f32: 10000.0f)).f32
  )
  if metadata.contains("llama.attention.head_count_kv"):
    config.numberOfKeyValueHeads = metadata["llama.attention.head_count_kv"].getInt()
  else:
    config.numberOfKeyValueHeads = config.numberOfHeads

  if contextLength > 0:
    config.contextLength = contextLength

  config.headSize = config.dim div config.numberOfHeads

  let weights = Weights()
  let ti = gguf.tensorInfos
  let doff = gguf.tensorDataOffset

  weights.tokenEmbeddingTable = loadQuantized(ti["token_embd.weight"], m, doff)

  weights.rmsAttWeight = newSeq[seq[float32]](config.numberOfLayers)
  weights.wq = newSeq[FloatTensor](config.numberOfLayers)
  weights.wk = newSeq[FloatTensor](config.numberOfLayers)
  weights.wv = newSeq[FloatTensor](config.numberOfLayers)
  weights.wo = newSeq[FloatTensor](config.numberOfLayers)
  weights.rmsFfnWeight = newSeq[seq[float32]](config.numberOfLayers)
  weights.w1 = newSeq[FloatTensor](config.numberOfLayers)
  weights.w2 = newSeq[FloatTensor](config.numberOfLayers)
  weights.w3 = newSeq[FloatTensor](config.numberOfLayers)

  for l in 0..<config.numberOfLayers:
    weights.rmsAttWeight[l] = toFloatSeq(ti["blk." & $l & ".attn_norm.weight"], m, doff)
    weights.wq[l] = loadQuantized(ti["blk." & $l & ".attn_q.weight"], m, doff)
    weights.wk[l] = loadQuantized(ti["blk." & $l & ".attn_k.weight"], m, doff)
    weights.wv[l] = loadQuantized(ti["blk." & $l & ".attn_v.weight"], m, doff)
    weights.wo[l] = loadQuantized(ti["blk." & $l & ".attn_output.weight"], m, doff)
    weights.rmsFfnWeight[l] = toFloatSeq(ti["blk." & $l & ".ffn_norm.weight"], m, doff)
    weights.w1[l] = loadQuantized(ti["blk." & $l & ".ffn_gate.weight"], m, doff)
    weights.w2[l] = loadQuantized(ti["blk." & $l & ".ffn_down.weight"], m, doff)
    weights.w3[l] = loadQuantized(ti["blk." & $l & ".ffn_up.weight"], m, doff)

  weights.rmsFinalWeight = toFloatSeq(ti["output_norm.weight"], m, doff)

  if ti.contains("output.weight"):
    weights.wcls = loadQuantized(ti["output.weight"], m, doff)
  else:
    weights.wcls = weights.tokenEmbeddingTable

  let ropeScaling = ti.contains("rope_freqs")
  let (cr, ci) = precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                                   ropeScaling, 8.0f32, 1.0f32, 3.0f32, 8192.0f32)
  weights.freqCisReal = cr
  weights.freqCisImag = ci

  return LlamaModel(config: config, tokenizer: tokenizer, weights: weights, memFile: m)
