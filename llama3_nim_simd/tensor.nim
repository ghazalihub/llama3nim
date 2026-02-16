import std/[math, sequtils, sugar, os, cpuinfo]
import gguf
import nimsimd/[avx2, sse2, fma]
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

proc hf_to_f32*(h: uint16): float32 {.inline.} =
  let h32 = h.uint32
  let s = (h32 shr 15) shl 31
  var e = (h32 shr 10) and 0x1F
  var m = h32 and 0x03FF
  if e == 0:
    if m != 0:
      while (m and 0x0400) == 0:
        m = m shl 1
        e -= 1
      e += 1
      m = m and 0x03FF
      let res = s or ((e + (127 - 15)) shl 23) or (m shl 13)
      return cast[float32](res)
    return cast[float32](s)
  if e == 31:
    let res = s or 0x7F800000'u32 or (m shl 13)
    return cast[float32](res)
  let res = s or ((e + (127 - 15)) shl 23) or (m shl 13)
  return cast[float32](res)

proc getFloat*(t: FloatTensor, i: int): float32 {.inline.} =
  case t.kind:
  of fkArray: return t.data[i]
  of fkQ4_0:
    let blockIndex = i div 32
    let blockOffset = blockIndex * 18
    let scaleBits = cast[ptr uint16](addr t.storage4[blockOffset])[]
    let scale = hf_to_f32(scaleBits)
    let modIndex = i mod 32
    var quant: byte
    if modIndex < 16:
      quant = t.storage4[blockOffset + 2 + modIndex] and 0x0F
    else:
      quant = (t.storage4[blockOffset + 2 + modIndex - 16] shr 4) and 0x0F
    return (quant.float32 - 8.0f32) * scale
  of fkQ8_0:
    let blockIndex = i div 32
    let blockOffset = blockIndex * 34
    let scaleBits = cast[ptr uint16](addr t.storage8[blockOffset])[]
    let scale = hf_to_f32(scaleBits)
    let quant = cast[int8](t.storage8[blockOffset + 2 + (i mod 32)])
    return quant.float32 * scale

proc setFloat*(t: FloatTensor, i: int, v: float32) {.inline.} =
  case t.kind:
  of fkArray: t.data[i] = v
  else: discard

proc newArrayFloatTensor*(size: int): FloatTensor =
  result = FloatTensor(kind: fkArray, size: size, data: newSeq[float32](size))

proc dotAvx(a, b: ptr float32, size: int): float32 {.inline.} =
  var sumv0 = mm256_setzero_ps()
  var sumv1 = mm256_setzero_ps()
  var i = 0
  while i <= size - 16:
    sumv0 = mm256_fmadd_ps(mm256_loadu_ps(cast[ptr float32](cast[int](a) + (i + 0) * 4)), mm256_loadu_ps(cast[ptr float32](cast[int](b) + (i + 0) * 4)), sumv0)
    sumv1 = mm256_fmadd_ps(mm256_loadu_ps(cast[ptr float32](cast[int](a) + (i + 8) * 4)), mm256_loadu_ps(cast[ptr float32](cast[int](b) + (i + 8) * 4)), sumv1)
    i += 16

  sumv0 = mm256_add_ps(sumv0, sumv1)

  while i <= size - 8:
    sumv0 = mm256_fmadd_ps(mm256_loadu_ps(cast[ptr float32](cast[int](a) + i * 4)), mm256_loadu_ps(cast[ptr float32](cast[int](b) + i * 4)), sumv0)
    i += 8

  var res: array[8, float32]
  mm256_storeu_ps(addr res[0], sumv0)
  result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]

  while i < size:
    result += cast[ptr UncheckedArray[float32]](a)[i] * cast[ptr UncheckedArray[float32]](b)[i]
    i += 1

proc dotQ8Array(thiz: ptr byte, that: ptr float32, size: int): float32 {.inline.} =
  var i = 0
  result = 0.0f32
  while i <= size - 32:
    let scaleBits = cast[ptr uint16](cast[int](thiz) + (i div 32) * 34)[]
    let scale = hf_to_f32(scaleBits)
    let pWeights = cast[ptr int8](cast[int](thiz) + (i div 32) * 34 + 2)
    let pActs = cast[ptr float32](cast[int](that) + i * 4)

    var blockSumv = mm256_setzero_ps()
    for k in 0..3:
      let w8 = mm_loadl_epi64(cast[ptr M128i](cast[int](pWeights) + k * 8))
      let w32 = mm256_cvtepi8_epi32(w8)
      let wf = mm256_cvtepi32_ps(w32)
      let af = mm256_loadu_ps(cast[ptr float32](cast[int](pActs) + k * 8 * 4))
      blockSumv = mm256_fmadd_ps(wf, af, blockSumv)

    var res: array[8, float32]
    mm256_storeu_ps(addr res[0], blockSumv)
    result += (res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]) * scale
    i += 32

  while i < size:
    let blockIndex = i div 32
    let blockOffset = blockIndex * 34
    let scale = hf_to_f32(cast[ptr uint16](cast[int](thiz) + blockOffset)[])
    let quant = cast[int8](cast[ptr byte](cast[int](thiz) + blockOffset + 2 + (i mod 32))[])
    result += quant.float32 * scale * cast[ptr UncheckedArray[float32]](that)[i]
    i += 1

proc dotQ4Array(thiz: ptr byte, that: ptr float32, size: int): float32 {.inline.} =
  var i = 0
  result = 0.0f32
  let eight = mm256_set1_ps(8.0f32)
  let mask = mm_set1_epi8(0x0F)

  while i <= size - 32:
    let scaleBits = cast[ptr uint16](cast[int](thiz) + (i div 32) * 18)[]
    let scale = hf_to_f32(scaleBits)
    let pWeights = cast[ptr byte](cast[int](thiz) + (i div 32) * 18 + 2)
    let pActs = cast[ptr float32](cast[int](that) + i * 4)

    let w16 = mm_loadu_si128(pWeights)
    let lo8 = mm_and_si128(w16, mask)
    let hi8 = mm_and_si128(mm_srli_epi16(w16, 4), mask)

    var blockSumv = mm256_setzero_ps()

    # Low quants (first 16)
    let w0_8 = mm256_cvtepi8_epi32(lo8)
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w0_8), eight), mm256_loadu_ps(pActs), blockSumv)
    let w1_8 = mm256_cvtepi8_epi32(mm_unpackhi_epi64(lo8, lo8))
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w1_8), eight), mm256_loadu_ps(cast[ptr float32](cast[int](pActs) + 32)), blockSumv)

    # High quants (next 16)
    let w2_8 = mm256_cvtepi8_epi32(hi8)
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w2_8), eight), mm256_loadu_ps(cast[ptr float32](cast[int](pActs) + 64)), blockSumv)
    let w3_8 = mm256_cvtepi8_epi32(mm_unpackhi_epi64(hi8, hi8))
    blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w3_8), eight), mm256_loadu_ps(cast[ptr float32](cast[int](pActs) + 96)), blockSumv)

    var res: array[8, float32]
    mm256_storeu_ps(addr res[0], blockSumv)
    result += (res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]) * scale
    i += 32

  while i < size:
    let blockIndex = i div 32
    let blockOffset = blockIndex * 18
    let scale = hf_to_f32(cast[ptr uint16](cast[int](thiz) + blockOffset)[])
    let modIndex = i mod 32
    var quant: byte
    let storage = cast[ptr UncheckedArray[byte]](cast[int](thiz) + blockOffset + 2)
    if modIndex < 16:
      quant = storage[modIndex] and 0x0F
    else:
      quant = (storage[modIndex - 16] shr 4) and 0x0F
    result += (quant.float32 - 8.0f32) * scale * cast[ptr UncheckedArray[float32]](that)[i]
    i += 1

proc dot*(thiz, that: FloatTensor, thisOffset, thatOffset, size: int): float32 {.inline.} =
  if thiz.kind == fkArray and that.kind == fkArray:
    return dotAvx(addr thiz.data[thisOffset], addr that.data[thatOffset], size)
  elif thiz.kind == fkQ8_0 and that.kind == fkArray:
    return dotQ8Array(cast[ptr byte](cast[int](thiz.storage8) + (thisOffset div 32) * 34), addr that.data[thatOffset], size)
  elif thiz.kind == fkQ4_0 and that.kind == fkArray:
    return dotQ4Array(cast[ptr byte](cast[int](thiz.storage4) + (thisOffset div 32) * 18), addr that.data[thatOffset], size)
  else:
    result = 0.0f32
    for j in 0..<size:
      result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j)

proc matmulRowWorker(kind: FloatTensorKind, pThiz, pThat, pOut: pointer, sR, eR, dim1: int) =
  let pThatF = cast[ptr float32](pThat)
  let pOutF = cast[ptr float32](pOut)
  case kind:
  of fkArray:
    let pThizF = cast[ptr float32](pThiz)
    for i in sR..<eR:
      cast[ptr UncheckedArray[float32]](pOutF)[i] = dotAvx(cast[ptr float32](cast[int](pThizF) + i * dim1 * 4), pThatF, dim1)
  of fkQ8_0:
    let pThizB = cast[ptr byte](pThiz)
    for i in sR..<eR:
      cast[ptr UncheckedArray[float32]](pOutF)[i] = dotQ8Array(cast[ptr byte](cast[int](pThizB) + (i * dim1 div 32) * 34), pThatF, dim1)
  of fkQ4_0:
    let pThizB = cast[ptr byte](pThiz)
    for i in sR..<eR:
      cast[ptr UncheckedArray[float32]](pOutF)[i] = dotQ4Array(cast[ptr byte](cast[int](pThizB) + (i * dim1 div 32) * 18), pThatF, dim1)

proc matmul*(thiz, that, `out`: FloatTensor, dim0, dim1: int) =
  var ctx = malebolgia.createMaster()
  let numThreads = countProcessors()
  let chunkSize = (dim0 + numThreads - 1) div numThreads

  let pThiz = case thiz.kind:
    of fkArray: cast[pointer](addr thiz.data[0])
    of fkQ4_0: cast[pointer](thiz.storage4)
    of fkQ8_0: cast[pointer](thiz.storage8)
  let pThat = cast[pointer](addr that.data[0])
  let pOut = cast[pointer](addr `out`.data[0])
  let kind = thiz.kind

  ctx.awaitAll:
    for t in 0..<numThreads:
      let startRow = t * chunkSize
      let endRow = min(startRow + chunkSize, dim0)
      if startRow >= endRow: break
      ctx.spawn matmulRowWorker(kind, pThiz, pThat, pOut, startRow, endRow, dim1)

proc copyTo*(thiz: FloatTensor, thisOffset: int, that: FloatTensor, thatOffset: int, size: int) {.inline.} =
  if thiz.kind == fkArray and that.kind == fkArray:
    copyMem(addr that.data[thatOffset], addr thiz.data[thisOffset], size * 4)
  else:
    for i in 0..<size:
      that.setFloat(thatOffset + i, thiz.getFloat(thisOffset + i))

proc fillInPlace*(t: FloatTensor, offset, size: int, val: float32) {.inline.} =
  if t.kind == fkArray:
    for i in offset..<offset+size: t.data[i] = val
  else:
    for i in offset..<offset+size: t.setFloat(i, val)

proc addInPlace*(t, that: FloatTensor) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let size = t.size
    let d1 = addr t.data[0]
    let d2 = addr that.data[0]
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(cast[ptr float32](cast[int](d1) + i * 4), mm256_add_ps(mm256_loadu_ps(cast[ptr float32](cast[int](d1) + i * 4)), mm256_loadu_ps(cast[ptr float32](cast[int](d2) + i * 4))))
      i += 8
    while i < size:
      t.data[i] += that.data[i]
      i += 1
  else:
    for i in 0..<t.size: setFloat(t, i, getFloat(t, i) + getFloat(that, i))

proc multiplyInPlace*(t, that: FloatTensor) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let size = t.size
    let d1 = addr t.data[0]
    let d2 = addr that.data[0]
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(cast[ptr float32](cast[int](d1) + i * 4), mm256_mul_ps(mm256_loadu_ps(cast[ptr float32](cast[int](d1) + i * 4)), mm256_loadu_ps(cast[ptr float32](cast[int](d2) + i * 4))))
      i += 8
    while i < size:
      t.data[i] *= that.data[i]
      i += 1
  else:
    for i in 0..<t.size: setFloat(t, i, getFloat(t, i) * getFloat(that, i))

proc divideInPlace*(t: FloatTensor, offset, size: int, val: float32) {.inline.} =
  if t.kind == fkArray:
    let d = addr t.data[offset]
    let v = mm256_set1_ps(val)
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(cast[ptr float32](cast[int](d) + i * 4), mm256_div_ps(mm256_loadu_ps(cast[ptr float32](cast[int](d) + i * 4)), v))
      i += 8
    while i < size:
      t.data[offset + i] /= val
      i += 1
  else:
    for i in offset..<offset+size: t.setFloat(i, getFloat(t, i) / val)

proc softmaxInPlace*(t: FloatTensor, offset, size: int) =
  var maxVal = getFloat(t, offset)
  for i in 1..<size:
    let v = getFloat(t, offset + i)
    if v > maxVal: maxVal = v
  var sum = 0.0f32
  for i in 0..<size:
    let v = exp(getFloat(t, offset + i) - maxVal)
    setFloat(t, offset + i, v)
    sum += v
  divideInPlace(t, offset, size, sum)

proc saxpyInPlace*(t: FloatTensor, offset: int, that: FloatTensor, thatOffset: int, size: int, a: float32) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let d1 = addr t.data[offset]
    let d2 = addr that.data[thatOffset]
    let av = mm256_set1_ps(a)
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(cast[ptr float32](cast[int](d1) + i * 4), mm256_fmadd_ps(av, mm256_loadu_ps(cast[ptr float32](cast[int](d2) + i * 4)), mm256_loadu_ps(cast[ptr float32](cast[int](d1) + i * 4))))
      i += 8
    while i < size:
      t.data[offset + i] += a * that.data[thatOffset + i]
      i += 1
  else:
    for i in 0..<size: setFloat(t, offset + i, a * getFloat(that, thatOffset + i) + getFloat(t, offset + i))

proc argmax*(t: FloatTensor, offset, size: int): int =
  var maxIndex = offset
  var maxVal = getFloat(t, offset)
  for i in 1..<size:
    let v = getFloat(t, offset + i)
    if v > maxVal:
      maxVal = v
      maxIndex = offset + i
  return maxIndex

proc argmax*(t: FloatTensor): int = return argmax(t, 0, t.size)

proc rmsnorm*(outT, x: FloatTensor, weight: seq[float32], size: int, eps: float32) =
  var ss = 0.0f32
  if x.kind == fkArray:
    let d = addr x.data[0]
    var sumv = mm256_setzero_ps()
    var i = 0
    while i <= size - 8:
      let v = mm256_loadu_ps(cast[ptr float32](cast[int](d) + i * 4))
      sumv = mm256_fmadd_ps(v, v, sumv)
      i += 8
    var res: array[8, float32]
    mm256_storeu_ps(addr res[0], sumv)
    ss = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]
    while i < size:
      ss += x.data[i] * x.data[i]
      i += 1
  else:
    for i in 0..<size: ss += getFloat(x, i) * getFloat(x, i)

  ss /= size.float32
  ss += eps
  ss = 1.0f32 / sqrt(ss)

  if outT.kind == fkArray and x.kind == fkArray:
    let d_out = addr outT.data[0]
    let d_in = addr x.data[0]
    let sv = mm256_set1_ps(ss)
    var i = 0
    while i <= size - 8:
      mm256_storeu_ps(cast[ptr float32](cast[int](d_out) + i * 4), mm256_mul_ps(mm256_loadu_ps(addr weight[i]), mm256_mul_ps(sv, mm256_loadu_ps(cast[ptr float32](cast[int](d_in) + i * 4)))))
      i += 8
    while i < size:
      outT.data[i] = weight[i] * (ss * x.data[i])
      i += 1
  else:
    for i in 0..<size: setFloat(outT, i, weight[i] * (ss * getFloat(x, i)))
