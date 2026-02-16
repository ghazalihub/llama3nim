import std/[math, sequtils, sugar]
import gguf
import nimsimd/avx2

type
  FloatTensorKind* = enum
    fkArray, fkQ4_0, fkQ8_0

  FloatTensor* = ref object
    size*: int
    case kind*: FloatTensorKind
    of fkArray: data*: seq[float32]
    of fkQ4_0: storage4*: ptr UncheckedArray[byte]
    of fkQ8_0: storage8*: ptr UncheckedArray[byte]

proc float16ToFloat32*(h: uint16): float32 {.inline.} =
  let
    s = (h shr 15) and 0x1
    e = (h shr 10) and 0x1f
    m = h and 0x03ff

  if e == 0:
    if m == 0:
      return if s == 1: -0.0f32 else: 0.0f32
    else:
      return (if s == 1: -1.0f32 else: 1.0f32) * (m.float32 * pow(2.0f32, -24.0f32))
  elif e == 31:
    if m == 0:
      return if s == 1: -Inf else: Inf
    else:
      return NaN

  return (if s == 1: -1.0f32 else: 1.0f32) * pow(2.0f32, e.float32 - 15.0f32) * (1.0f32 + m.float32 / 1024.0f32)

proc getFloat*(t: FloatTensor, i: int): float32 {.inline.} =
  case t.kind:
  of fkArray: return t.data[i]
  of fkQ4_0:
    let blockIndex = i div 32
    let blockOffset = blockIndex * 18
    let scaleBits = cast[ptr uint16](addr t.storage4[blockOffset])[]
    let scale = float16ToFloat32(scaleBits)
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
    let scale = float16ToFloat32(scaleBits)
    let quant = cast[int8](t.storage8[blockOffset + 2 + (i mod 32)])
    return quant.float32 * scale

proc setFloat*(t: FloatTensor, i: int, v: float32) {.inline.} =
  case t.kind:
  of fkArray: t.data[i] = v
  else: discard

proc newArrayFloatTensor*(size: int): FloatTensor =
  result = FloatTensor(kind: fkArray, size: size, data: newSeq[float32](size))

proc dotAvx(a, b: ptr float32, size: int): float32 =
  var sumv = mm256_setzero_ps()
  var i = 0
  while i <= size - 8:
    let av = mm256_loadu_ps(cast[ptr float32](cast[int](a) + i * 4))
    let bv = mm256_loadu_ps(cast[ptr float32](cast[int](b) + i * 4))
    sumv = mm256_add_ps(sumv, mm256_mul_ps(av, bv))
    i += 8

  var res: array[8, float32]
  mm256_storeu_ps(addr res[0], sumv)
  result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]

  while i < size:
    result += cast[ptr UncheckedArray[float32]](a)[i] * cast[ptr UncheckedArray[float32]](b)[i]
    i += 1

proc dot*(thiz, that: FloatTensor, thisOffset, thatOffset, size: int): float32 =
  if thiz.kind == fkArray and that.kind == fkArray:
    return dotAvx(addr thiz.data[thisOffset], addr that.data[thatOffset], size)

  result = 0.0f32
  if thiz.kind == fkQ8_0 and that.kind == fkArray:
    let blockSize = 32
    var i = 0
    while i <= size - blockSize:
      let blockIndex = (thisOffset + i) div blockSize
      let blockOffset = blockIndex * 34
      let scaleBits = cast[ptr uint16](addr thiz.storage8[blockOffset])[]
      let scale = float16ToFloat32(scaleBits)

      var blockSumv = mm256_setzero_ps()
      let pWeights = cast[ptr M128i](addr thiz.storage8[blockOffset + 2])
      let pActs = cast[ptr float32](addr that.data[thatOffset + i])

      for k in 0..3:
        # Load 8 bytes of weights, convert to 8 floats
        # Actually mm256_cvtepi8_epi32 needs a 128-bit input for 8 elements
        # mm_loadl_epi64 loads 8 bytes into lower 64 bits of 128-bit reg
        let w8 = mm_loadl_epi64(cast[ptr M128i](cast[int](pWeights) + k * 8))
        let w32 = mm256_cvtepi8_epi32(w8)
        let wf = mm256_cvtepi32_ps(w32)
        let af = mm256_loadu_ps(cast[ptr float32](cast[int](pActs) + k * 8 * 4))
        blockSumv = mm256_add_ps(blockSumv, mm256_mul_ps(wf, af))

      var res: array[8, float32]
      mm256_storeu_ps(addr res[0], blockSumv)
      result += (res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]) * scale
      i += blockSize

    while i < size:
      result += thiz.getFloat(thisOffset + i) * that.getFloat(thatOffset + i)
      i += 1
  elif thiz.kind == fkQ4_0 and that.kind == fkArray:
    # Vectorizing Q4_0 is a bit more complex due to unpacking
    # For now, keep it optimized scalar or semi-vectorized
    let blockSize = 32
    var i = 0
    while i <= size - blockSize:
      let blockIndex = (thisOffset + i) div blockSize
      let blockOffset = blockIndex * 18
      let scaleBits = cast[ptr uint16](addr thiz.storage4[blockOffset])[]
      let scale = float16ToFloat32(scaleBits)

      var blockSum = 0.0f32
      for j in 0..<16:
        let b = thiz.storage4[blockOffset + 2 + j]
        let q0 = (b and 0x0F).float32 - 8.0f32
        let q1 = (b shr 4).float32 - 8.0f32
        blockSum += q0 * that.data[thatOffset + i + j]
        blockSum += q1 * that.data[thatOffset + i + j + 16]
      result += blockSum * scale
      i += blockSize
    while i < size:
      result += thiz.getFloat(thisOffset + i) * that.getFloat(thatOffset + i)
      i += 1
  else:
    for j in 0..<size:
      result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j)

proc matmul*(thiz, that, `out`: FloatTensor, dim0, dim1: int) =
  for i in 0..<dim0:
    `out`.setFloat(i, dot(thiz, that, i * dim1, 0, dim1))

proc copyTo*(thiz: FloatTensor, thisOffset: int, that: FloatTensor, thatOffset: int, size: int) =
  for i in 0..<size:
    that.setFloat(thatOffset + i, thiz.getFloat(thisOffset + i))

proc fillInPlace*(t: FloatTensor, offset, size: int, val: float32) =
  for i in offset..<offset+size:
    t.setFloat(i, val)

proc addInPlace*(t, that: FloatTensor) =
  for i in 0..<t.size:
    setFloat(t, i, getFloat(t, i) + getFloat(that, i))

proc multiplyInPlace*(t, that: FloatTensor) =
  for i in 0..<t.size:
    setFloat(t, i, getFloat(t, i) * getFloat(that, i))

proc divideInPlace*(t: FloatTensor, offset, size: int, val: float32) =
  for i in offset..<offset+size:
    setFloat(t, i, getFloat(t, i) / val)

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
  for i in 0..<size:
    setFloat(t, offset + i, getFloat(t, offset + i) / sum)

proc saxpyInPlace*(t: FloatTensor, offset: int, that: FloatTensor, thatOffset: int, size: int, a: float32) =
  for i in 0..<size:
    setFloat(t, offset + i, a * getFloat(that, thatOffset + i) + getFloat(t, offset + i))

proc argmax*(t: FloatTensor, offset, size: int): int =
  var maxIndex = offset
  var maxVal = getFloat(t, offset)
  for i in 1..<size:
    let v = getFloat(t, offset + i)
    if v > maxVal:
      maxVal = v
      maxIndex = offset + i
  return maxIndex

proc argmax*(t: FloatTensor): int =
  return argmax(t, 0, t.size)

proc rmsnorm*(outT, x: FloatTensor, weight: seq[float32], size: int, eps: float32) =
  var ss = 0.0f32
  for i in 0..<size:
    let xi = getFloat(x, i)
    ss += xi * xi
  ss /= size.float32
  ss += eps
  ss = 1.0f32 / sqrt(ss)
  for i in 0..<size:
    setFloat(outT, i, weight[i] * (ss * getFloat(x, i)))
