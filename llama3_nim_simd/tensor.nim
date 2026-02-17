import std/[math, cpuinfo]
import nimsimd/[avx2, sse2, fma, f16c]
import malebolgia

type
  FloatTensorKind* = enum
    fkArray, fkQ4_0, fkQ8_0

  FloatTensor* = ref object
    size*: int
    case kind*: FloatTensorKind
    of fkArray: data*: seq[float32]
    of fkQ4_0: storage4*: ptr UncheckedArray[byte]
    of fkQ8_0: storage8*: ptr UncheckedArray[byte]

proc hsum(v: M256): float32 {.inline.} =
  let v128 = mm_add_ps(mm256_extractf128_ps(v, 1), mm256_castps256_ps128(v))
  let shuf = mm_movehdup_ps(v128)
  let sums = mm_add_ps(v128, shuf)
  let shuf2 = mm_movehl_ps(sums, sums)
  let sums2 = mm_add_ss(sums, shuf2)
  return mm_cvtss_f32(sums2)

proc dotAvx*(a, b: ptr float32, size: int): float32 {.inline.} =
  var sumv = mm256_setzero_ps()
  var i = 0
  let pa = cast[ptr UncheckedArray[float32]](a)
  let pb = cast[ptr UncheckedArray[float32]](b)
  while i <= size - 8:
    sumv = mm256_fmadd_ps(mm256_loadu_ps(addr pa[i]), mm256_loadu_ps(addr pb[i]), sumv)
    i += 8
  result = hsum(sumv)
  while i < size:
    result += pa[i] * pb[i]
    i += 1

proc dotQ8Array*(pW: ptr byte, pA: ptr float32, size: int): float32 {.inline.} =
  var i = 0
  result = 0.0f32
  let pa = cast[ptr UncheckedArray[float32]](pA)
  while i <= size - 32:
    let blockIdx = i div 32
    let pBlock = cast[ptr byte](cast[uint](pW) + blockIdx.uint * 34)
    let h = mm_cvtsi32_si128(cast[ptr int32](pBlock)[])
    let f = mm256_cvtph_ps(h)
    let scale = mm_cvtss_f32(mm256_castps256_ps128(f))
    let pWeights = cast[ptr int8](cast[uint](pBlock) + 2)
    var blockSumv = mm256_setzero_ps()
    for k in 0..3:
      let w8 = mm_loadl_epi64(cast[ptr M128i](cast[uint](pWeights) + k.uint * 8))
      let wf = mm256_cvtepi32_ps(mm256_cvtepi8_epi32(w8))
      let af = mm256_loadu_ps(addr pa[i + k * 8])
      blockSumv = mm256_fmadd_ps(wf, af, blockSumv)
    result += hsum(blockSumv) * scale
    i += 32

proc dotQ4Array*(pW: ptr byte, pA: ptr float32, size: int): float32 {.inline.} =
  var i = 0
  result = 0.0f32
  let eight = mm256_set1_ps(8.0f32)
  let mask = mm_set1_epi8(0x0F)
  let pa = cast[ptr UncheckedArray[float32]](pA)
  while i <= size - 32:
    let blockIdx = i div 32
    let pBlock = cast[ptr byte](cast[uint](pW) + blockIdx.uint * 18)
    let h = mm_cvtsi32_si128(cast[ptr int32](pBlock)[])
    let f = mm256_cvtph_ps(h)
    let scale = mm_cvtss_f32(mm256_castps256_ps128(f))
    let pWeights = cast[ptr byte](cast[uint](pBlock) + 2)
    let w16 = mm_loadu_si128(pWeights)
    let lo8 = mm_and_si128(w16, mask)
    let hi8 = mm_and_si128(mm_srli_epi16(w16, 4), mask)
    var blockSumv = mm256_setzero_ps()
    let w0_8 = mm256_cvtepi8_epi32(lo8)
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w0_8), eight), mm256_loadu_ps(addr pa[i]), blockSumv)
    let w1_8 = mm256_cvtepi8_epi32(mm_unpackhi_epi64(lo8, lo8))
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w1_8), eight), mm256_loadu_ps(addr pa[i + 8]), blockSumv)
    let w2_8 = mm256_cvtepi8_epi32(hi8)
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w2_8), eight), mm256_loadu_ps(addr pa[i + 16]), blockSumv)
    let w3_8 = mm256_cvtepi8_epi32(mm_unpackhi_epi64(hi8, hi8))
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w3_8), eight), mm256_loadu_ps(addr pa[i + 24]), blockSumv)
    result += hsum(blockSumv) * scale
    i += 32

proc gemv_array_array_worker(pW, pA, pO: ptr float32, sR, eR, dim1: int) =
  let poa = cast[ptr UncheckedArray[float32]](pO)
  let paa = cast[ptr UncheckedArray[float32]](pA)
  var r = sR
  while r <= eR - 4:
    var acc0 = mm256_setzero_ps(); var acc1 = mm256_setzero_ps()
    var acc2 = mm256_setzero_ps(); var acc3 = mm256_setzero_ps()
    let pW0 = cast[ptr UncheckedArray[float32]](cast[uint](pW) + (r + 0).uint * dim1.uint * 4)
    let pW1 = cast[ptr UncheckedArray[float32]](cast[uint](pW) + (r + 1).uint * dim1.uint * 4)
    let pW2 = cast[ptr UncheckedArray[float32]](cast[uint](pW) + (r + 2).uint * dim1.uint * 4)
    let pW3 = cast[ptr UncheckedArray[float32]](cast[uint](pW) + (r + 3).uint * dim1.uint * 4)
    var c = 0
    while c <= dim1 - 8:
      let av = mm256_loadu_ps(addr paa[c])
      acc0 = mm256_fmadd_ps(mm256_loadu_ps(addr pW0[c]), av, acc0)
      acc1 = mm256_fmadd_ps(mm256_loadu_ps(addr pW1[c]), av, acc1)
      acc2 = mm256_fmadd_ps(mm256_loadu_ps(addr pW2[c]), av, acc2)
      acc3 = mm256_fmadd_ps(mm256_loadu_ps(addr pW3[c]), av, acc3)
      c += 8
    poa[r + 0] = hsum(acc0); poa[r + 1] = hsum(acc1); poa[r + 2] = hsum(acc2); poa[r + 3] = hsum(acc3)
    while c < dim1:
      let av = paa[c]
      poa[r + 0] += pW0[c] * av; poa[r + 1] += pW1[c] * av; poa[r + 2] += pW2[c] * av; poa[r + 3] += pW3[c] * av
      c += 1
    r += 4
  while r < eR:
    poa[r] = dotAvx(cast[ptr float32](cast[uint](pW) + r.uint * dim1.uint * 4), pA, dim1)
    r += 1

proc gemv_q8_array_worker(pW: ptr byte, pA, pO: ptr float32, sR, eR, dim1: int) =
  let poa = cast[ptr UncheckedArray[float32]](pO)
  for r in sR..<eR:
    poa[r] = dotQ8Array(cast[ptr byte](cast[uint](pW) + (r.uint * dim1.uint div 32) * 34), pA, dim1)

proc gemv_q4_array_worker(pW: ptr byte, pA, pO: ptr float32, sR, eR, dim1: int) =
  let poa = cast[ptr UncheckedArray[float32]](pO)
  for r in sR..<eR:
    poa[r] = dotQ4Array(cast[ptr byte](cast[uint](pW) + (r.uint * dim1.uint div 32) * 18), pA, dim1)

proc matmulMT*(ctx: var Master, thiz, that, res: FloatTensor, dim0, dim1: int) =
  let numThreads = countProcessors()
  let chunkSize = (dim0 + numThreads - 1) div numThreads
  let pA = addr that.data[0]
  let pO = addr res.data[0]

  case thiz.kind:
  of fkArray:
    let pW = cast[ptr float32](addr thiz.data[0])
    ctx.awaitAll:
      for t in 0..<numThreads:
        let sR = t * chunkSize; let eR = min(sR + chunkSize, dim0)
        if sR < eR: ctx.spawn gemv_array_array_worker(pW, pA, pO, sR, eR, dim1)
  of fkQ8_0:
    let pW = cast[ptr byte](thiz.storage8)
    ctx.awaitAll:
      for t in 0..<numThreads:
        let sR = t * chunkSize; let eR = min(sR + chunkSize, dim0)
        if sR < eR: ctx.spawn gemv_q8_array_worker(pW, pA, pO, sR, eR, dim1)
  of fkQ4_0:
    let pW = cast[ptr byte](thiz.storage4)
    ctx.awaitAll:
      for t in 0..<numThreads:
        let sR = t * chunkSize; let eR = min(sR + chunkSize, dim0)
        if sR < eR: ctx.spawn gemv_q4_array_worker(pW, pA, pO, sR, eR, dim1)

proc matmul*(thiz, that, res: FloatTensor, dim0, dim1: int) =
  var ctx = malebolgia.createMaster()
  matmulMT(ctx, thiz, that, res, dim0, dim1)

proc getFloat*(t: FloatTensor, i: int): float32 {.inline.} =
  case t.kind:
  of fkArray: return t.data[i]
  of fkQ4_0:
    let blockIndex = i div 32; let blockOffset = blockIndex * 18
    let h = mm_cvtsi32_si128(cast[ptr int32](addr t.storage4[blockOffset])[])
    let f = mm256_cvtph_ps(h); let scale = mm_cvtss_f32(mm256_castps256_ps128(f))
    let modIndex = i mod 32; var quant: byte
    if modIndex < 16: quant = t.storage4[blockOffset + 2 + modIndex] and 0x0F
    else: quant = (t.storage4[blockOffset + 2 + modIndex - 16] shr 4) and 0x0F
    return (quant.float32 - 8.0f32) * scale
  of fkQ8_0:
    let blockIndex = i div 32; let blockOffset = blockIndex * 34
    let h = mm_cvtsi32_si128(cast[ptr int32](addr t.storage8[blockOffset])[])
    let f = mm256_cvtph_ps(h); let scale = mm_cvtss_f32(mm256_castps256_ps128(f))
    let quant = cast[int8](t.storage8[blockOffset + 2 + (i mod 32)])
    return quant.float32 * scale

proc setFloat*(t: FloatTensor, i: int, v: float32) {.inline.} =
  case t.kind:
  of fkArray: t.data[i] = v
  else: discard

proc newArrayFloatTensor*(size: int): FloatTensor =
  let alignedSize = (size + 7) and not 7
  result = FloatTensor(kind: fkArray, size: size, data: newSeq[float32](alignedSize))

proc copyTo*(thiz: FloatTensor, thisOffset: int, that: FloatTensor, thatOffset: int, size: int) {.inline.} =
  if thiz.kind == fkArray and that.kind == fkArray:
    copyMem(addr that.data[thatOffset], addr thiz.data[thisOffset], size * 4)
  else:
    for i in 0..<size: that.setFloat(thatOffset + i, getFloat(thiz, thisOffset + i))

proc fillInPlace*(t: FloatTensor, offset, size: int, val: float32) {.inline.} =
  if t.kind == fkArray:
    for i in offset..<offset+size: t.data[i] = val
  else:
    for i in offset..<offset+size: setFloat(t, i, val)

proc addInPlace*(t, that: FloatTensor) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let size = t.size
    let d1 = cast[ptr UncheckedArray[float32]](addr t.data[0])
    let d2 = cast[ptr UncheckedArray[float32]](addr that.data[0])
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(addr d1[i], mm256_add_ps(mm256_loadu_ps(addr d1[i]), mm256_loadu_ps(addr d2[i]))); i += 8
    while i < size: d1[i] += d2[i]; i += 1

proc multiplyInPlace*(t, that: FloatTensor) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let size = t.size
    let d1 = cast[ptr UncheckedArray[float32]](addr t.data[0])
    let d2 = cast[ptr UncheckedArray[float32]](addr that.data[0])
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(addr d1[i], mm256_mul_ps(mm256_loadu_ps(addr d1[i]), mm256_loadu_ps(addr d2[i]))); i += 8
    while i < size: d1[i] *= d2[i]; i += 1

proc divideInPlace*(t: FloatTensor, offset, size: int, val: float32) {.inline.} =
  if t.kind == fkArray:
    let d = cast[ptr UncheckedArray[float32]](addr t.data[offset])
    let v = mm256_set1_ps(val); var i = 0
    while i <= size - 8:
      mm256_storeu_ps(addr d[i], mm256_div_ps(mm256_loadu_ps(addr d[i]), v)); i += 8
    while i < size: d[i] /= val; i += 1

proc softmaxInPlace*(t: FloatTensor, offset, size: int) =
  if t.kind == fkArray:
    let d = cast[ptr UncheckedArray[float32]](addr t.data[offset])
    var maxVal = d[0]
    for i in 1..<size:
      if d[i] > maxVal: maxVal = d[i]
    var sum = 0.0f32
    for i in 0..<size:
      d[i] = exp(d[i] - maxVal); sum += d[i]
    let vsum = mm256_set1_ps(sum); var i = 0
    while i <= size - 8:
      mm256_storeu_ps(addr d[i], mm256_div_ps(mm256_loadu_ps(addr d[i]), vsum)); i += 8
    while i < size: d[i] /= sum; i += 1

proc saxpyInPlace*(t: FloatTensor, offset: int, that: FloatTensor, thatOffset: int, size: int, a: float32) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let d1 = cast[ptr UncheckedArray[float32]](addr t.data[offset])
    let d2 = cast[ptr UncheckedArray[float32]](addr that.data[thatOffset])
    let av = mm256_set1_ps(a); var i = 0
    while i <= size - 8:
      mm256_storeu_ps(addr d1[i], mm256_fmadd_ps(av, mm256_loadu_ps(addr d2[i]), mm256_loadu_ps(addr d1[i]))); i += 8
    while i < size: d1[i] += a * d2[i]; i += 1

proc argmax*(t: FloatTensor, offset, size: int): int =
  var maxIndex = offset; var maxVal = getFloat(t, offset)
  for i in 1..<size:
    let v = getFloat(t, offset + i)
    if v > maxVal: maxVal = v; maxIndex = offset + i
  return maxIndex

proc argmax*(t: FloatTensor): int = return argmax(t, 0, t.size)

proc rmsnorm*(outT, x: FloatTensor, weight: seq[float32], size: int, eps: float32) =
  if outT.kind == fkArray and x.kind == fkArray:
    let d_in = cast[ptr UncheckedArray[float32]](addr x.data[0])
    let pw = cast[ptr UncheckedArray[float32]](unsafeAddr weight[0])
    var sumv = mm256_setzero_ps(); var i = 0
    while i <= size - 8:
      let v = mm256_loadu_ps(addr d_in[i]); sumv = mm256_fmadd_ps(v, v, sumv); i += 8
    var ss = hsum(sumv)
    while i < size: ss += d_in[i] * d_in[i]; i += 1
    ss = 1.0f32 / sqrt(ss / size.float32 + eps)
    let d_out = cast[ptr UncheckedArray[float32]](addr outT.data[0])
    let sv = mm256_set1_ps(ss); i = 0
    while i <= size - 8:
      mm256_storeu_ps(addr d_out[i], mm256_mul_ps(mm256_loadu_ps(addr pw[i]), mm256_mul_ps(sv, mm256_loadu_ps(addr d_in[i])))); i += 8
    while i < size: d_out[i] = pw[i] * (ss * d_in[i]); i += 1
