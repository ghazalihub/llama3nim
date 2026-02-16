import std/[math, sequtils, sugar]
import gguf

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

proc dot*(thiz, that: FloatTensor, thisOffset, thatOffset, size: int): float32 =
  result = 0.0f32
  if thiz.kind == fkArray and that.kind == fkArray:
    let d1 = thiz.data
    let d2 = that.data
    for j in 0..<size:
      result += d1[thisOffset + j] * d2[thatOffset + j]
  elif thiz.kind == fkQ4_0 and that.kind == fkArray:
    let d2 = that.data
    for j in 0..<size:
      result += thiz.getFloat(thisOffset + j) * d2[thatOffset + j]
  elif thiz.kind == fkQ8_0 and that.kind == fkArray:
    let d2 = that.data
    for j in 0..<size:
      result += thiz.getFloat(thisOffset + j) * d2[thatOffset + j]
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
