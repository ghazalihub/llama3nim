import std/[math, cpuinfo]
import nimsimd/[avx2, sse2, fma, f16c]
import malebolgia

type
  FloatTensorKind* = enum
    fkArray, fkQ4_0, fkQ8_0

  FloatTensor* = ref object
    size*: int
    case kind*: FloatTensorKind
    of fkArray:
      dataPtr*: ptr UncheckedArray[float32]
      isOwner*: bool
    of fkQ4_0: storage4*: ptr UncheckedArray[byte]
    of fkQ8_0: storage8*: ptr UncheckedArray[byte]

proc exp_ps*(x: M256): M256 {.inline.} =
  let tmp = mm256_mul_ps(x, mm256_set1_ps(1.44269504f32))
  let integer = mm256_cvttps_epi32(tmp)
  let fraction = mm256_sub_ps(tmp, mm256_cvtepi32_ps(integer))
  var p = mm256_set1_ps(0.05550411f32)
  p = mm256_fmadd_ps(p, fraction, mm256_set1_ps(0.24022650f32))
  p = mm256_fmadd_ps(p, fraction, mm256_set1_ps(0.69314718f32))
  p = mm256_fmadd_ps(p, fraction, mm256_set1_ps(1.0f32))
  let exponent = mm256_slli_epi32(mm256_add_epi32(integer, mm256_set1_epi32(127)), 23)
  result = mm256_mul_ps(p, cast[M256](exponent))

proc hsum*(v: M256): float32 {.inline.} =
  let v128 = mm_add_ps(mm256_extractf128_ps(v, 1), mm256_castps256_ps128(v))
  let shuf = mm_movehdup_ps(v128)
  let sums = mm_add_ps(v128, shuf)
  let shuf2 = mm_movehl_ps(sums, sums)
  let sums2 = mm_add_ss(sums, shuf2)
  return mm_cvtss_f32(sums2)

proc dotAvx*(a, b: ptr float32, size: int): float32 {.inline.} =
  var sumv = mm256_setzero_ps(); var i = 0
  let pa = cast[ptr UncheckedArray[float32]](a); let pb = cast[ptr UncheckedArray[float32]](b)
  while i <= size - 8:
    sumv = mm256_fmadd_ps(mm256_load_ps(addr pa[i]), mm256_load_ps(addr pb[i]), sumv); i += 8
  result = hsum(sumv)
  while i < size:
    result += pa[i] * pb[i]
    i += 1

proc dotQ8Array*(pW: ptr byte, pA: ptr float32, size: int): float32 {.inline.} =
  result = 0.0f32; let pa = cast[ptr UncheckedArray[float32]](pA); var currW = pW; var i = 0
  while i <= size - 32:
    let h = mm_cvtsi32_si128(cast[ptr int32](currW)[])
    let scale = mm_cvtss_f32(mm256_castps256_ps128(mm256_cvtph_ps(h)))
    let pWeights = cast[ptr int8](cast[uint](currW) + 2)
    var blockSumv = mm256_setzero_ps()
    for k in 0..3:
      let w8 = mm_loadl_epi64(cast[ptr M128i](cast[uint](pWeights) + k.uint * 8))
      let wf = mm256_cvtepi32_ps(mm256_cvtepi8_epi32(w8))
      blockSumv = mm256_fmadd_ps(wf, mm256_load_ps(addr pa[i + k * 8]), blockSumv)
    result += hsum(blockSumv) * scale; currW = cast[ptr byte](cast[uint](currW) + 34); i += 32

proc dotQ4Array*(pW: ptr byte, pA: ptr float32, size: int): float32 {.inline.} =
  result = 0.0f32; let eight = mm256_set1_ps(8.0f32); let mask = mm_set1_epi8(0x0F)
  let pa = cast[ptr UncheckedArray[float32]](pA); var currW = pW; var i = 0
  while i <= size - 32:
    let h = mm_cvtsi32_si128(cast[ptr int32](currW)[])
    let scale = mm_cvtss_f32(mm256_castps256_ps128(mm256_cvtph_ps(h)))
    let pWeights = cast[ptr byte](cast[uint](currW) + 2); let w16 = mm_loadu_si128(pWeights)
    let lo8 = mm_and_si128(w16, mask); let hi8 = mm_and_si128(mm_srli_epi16(w16, 4), mask)
    var blockSumv = mm256_setzero_ps()
    let w0_8 = mm256_cvtepi8_epi32(lo8); blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w0_8), eight), mm256_load_ps(addr pa[i]), blockSumv)
    let w1_8 = mm256_cvtepi8_epi32(mm_unpackhi_epi64(lo8, lo8)); blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w1_8), eight), mm256_load_ps(addr pa[i+8]), blockSumv)
    let w2_8 = mm256_cvtepi8_epi32(hi8); blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w2_8), eight), mm256_load_ps(addr pa[i+16]), blockSumv)
    let w3_8 = mm256_cvtepi8_epi32(mm_unpackhi_epi64(hi8, hi8)); blockSumv = mm256_fmadd_ps(mm256_sub_ps(mm256_cvtepi32_ps(w3_8), eight), mm256_load_ps(addr pa[i+24]), blockSumv)
    result += hsum(blockSumv) * scale; currW = cast[ptr byte](cast[uint](currW) + 18); i += 32

proc gemv_array_worker*(pW, pA, pO: ptr float32, sR, eR, dim1: int) =
  let poa = cast[ptr UncheckedArray[float32]](pO)
  for r in sR..<eR: poa[r] = dotAvx(cast[ptr float32](cast[uint](pW) + r.uint * dim1.uint * 4), pA, dim1)

proc gemv_q8_worker*(pW: ptr byte, pA, pO: ptr float32, sR, eR, dim1: int) =
  let poa = cast[ptr UncheckedArray[float32]](pO); let stride = (dim1 div 32) * 34
  for r in sR..<eR: poa[r] = dotQ8Array(cast[ptr byte](cast[uint](pW) + r.uint * stride.uint), pA, dim1)

proc gemv_q4_worker*(pW: ptr byte, pA, pO: ptr float32, sR, eR, dim1: int) =
  let poa = cast[ptr UncheckedArray[float32]](pO); let stride = (dim1 div 32) * 18
  for r in sR..<eR: poa[r] = dotQ4Array(cast[ptr byte](cast[uint](pW) + r.uint * stride.uint), pA, dim1)

proc matmulMT*(ctx: var Master, thiz, that, res: FloatTensor, dim0, dim1: int) =
  let numThreads = countProcessors(); let chunkSize = (dim0 + numThreads - 1) div numThreads
  let pA = cast[ptr float32](that.dataPtr); let pO = cast[ptr float32](res.dataPtr)
  ctx.awaitAll:
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

proc silu_worker*(pHb, pHb2: ptr float32, s, e: int) =
  let ph = cast[ptr UncheckedArray[float32]](pHb); let ph2 = cast[ptr UncheckedArray[float32]](pHb2)
  let onev = mm256_set1_ps(1.0f32); var i = s
  while i <= e - 8:
    let v = mm256_load_ps(addr ph[i]); let v2 = mm256_load_ps(addr ph2[i])
    let exp_v = exp_ps(mm256_sub_ps(mm256_setzero_ps(), v))
    let sig_v = mm256_div_ps(onev, mm256_add_ps(onev, exp_v))
    mm256_store_ps(addr ph[i], mm256_mul_ps(mm256_mul_ps(v, sig_v), v2)); i += 8
  while i < e:
    let v = ph[i]
    ph[i] = (v / (1.0f32 + exp(-v))) * ph2[i]
    i += 1

proc siluMultiplyMT*(ctx: var Master, hb, hb2: FloatTensor) =
  let size = hb.size; let numThreads = countProcessors(); let chunkSize = (size + numThreads - 1) div numThreads
  let pHb = cast[ptr float32](hb.dataPtr); let pHb2 = cast[ptr float32](hb2.dataPtr)
  ctx.awaitAll:
    for t in 0..<numThreads:
      let s = t * chunkSize; let e = min(s + chunkSize, size)
      if s < e: ctx.spawn silu_worker(pHb, pHb2, s, e)

proc getFloat*(t: FloatTensor, i: int): float32 {.inline.} =
  case t.kind:
  of fkArray: return t.dataPtr[i]
  of fkQ4_0:
    let blockIndex = i div 32; let blockOffset = blockIndex * 18
    let h = mm_cvtsi32_si128(cast[ptr int32](addr t.storage4[blockOffset])[])
    let scale = mm_cvtss_f32(mm256_castps256_ps128(mm256_cvtph_ps(h)))
    let modIndex = i mod 32; var quant: byte
    if modIndex < 16: quant = t.storage4[blockOffset + 2 + modIndex] and 0x0F
    else: quant = (t.storage4[blockOffset + 2 + modIndex - 16] shr 4) and 0x0F
    return (quant.float32 - 8.0f32) * scale
  of fkQ8_0:
    let blockIndex = i div 32; let blockOffset = blockIndex * 34
    let h = mm_cvtsi32_si128(cast[ptr int32](addr t.storage8[blockOffset])[])
    let scale = mm_cvtss_f32(mm256_castps256_ps128(mm256_cvtph_ps(h)))
    let quant = cast[int8](t.storage8[blockOffset + 2 + (i mod 32)])
    return quant.float32 * scale

proc setFloat*(t: FloatTensor, i: int, v: float32) {.inline.} =
  case t.kind:
  of fkArray: t.dataPtr[i] = v
  else: discard

when defined(linux):
  proc posix_memalign(memptr: ptr pointer, alignment: int, size: int): int {.header: "<stdlib.h>", importc: "posix_memalign".}
else:
  proc posix_memalign(memptr: ptr pointer, alignment: int, size: int): int =
    memptr[] = alloc0(size); return 0

proc newArrayFloatTensor*(size: int): FloatTensor =
  let alignedSize = (size + 7) and not 7
  var p: pointer
  if posix_memalign(addr p, 32, alignedSize * 4) != 0: raise newException(ValueError, "posix_memalign failed")
  zeroMem(p, alignedSize * 4)
  result = FloatTensor(kind: fkArray, size: size, dataPtr: cast[ptr UncheckedArray[float32]](p), isOwner: true)

proc free*(t: FloatTensor) =
  if t.kind == fkArray and t.isOwner and t.dataPtr != nil: dealloc(t.dataPtr); t.dataPtr = nil

proc copyTo*(thiz: FloatTensor, thisOffset: int, that: FloatTensor, thatOffset: int, size: int) =
  if thiz.kind == fkArray and that.kind == fkArray:
    copyMem(addr that.dataPtr[thatOffset], addr thiz.dataPtr[thisOffset], size * 4)
  elif that.kind == fkArray:
    let pd = cast[ptr UncheckedArray[float32]](addr that.dataPtr[thatOffset])
    if thiz.kind == fkQ8_0:
      var i = 0; var currW = cast[ptr byte](cast[uint](thiz.storage8) + (thisOffset div 32).uint * 34)
      while i <= size - 32:
        let h = mm_cvtsi32_si128(cast[ptr int32](currW)[])
        var scale = mm_cvtss_f32(mm256_castps256_ps128(mm256_cvtph_ps(h)))
        let scalev = mm256_broadcast_ss(addr scale)
        let pWeights = cast[ptr int8](cast[uint](currW) + 2)
        for k in 0..3:
          let w8 = mm_loadl_epi64(cast[ptr M128i](cast[uint](pWeights) + k.uint * 8))
          mm256_store_ps(addr pd[i + k * 8], mm256_mul_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(w8)), scalev))
        currW = cast[ptr byte](cast[uint](currW) + 34); i += 32
    elif thiz.kind == fkQ4_0:
      let eight = mm256_set1_ps(8.0f32); let mask = mm_set1_epi8(0x0F)
      var i = 0; var currW = cast[ptr byte](cast[uint](thiz.storage4) + (thisOffset div 32).uint * 18)
      while i <= size - 32:
        let h = mm_cvtsi32_si128(cast[ptr int32](currW)[])
        var scale = mm_cvtss_f32(mm256_castps256_ps128(mm256_cvtph_ps(h)))
        let scalev = mm256_broadcast_ss(addr scale)
        let pWeights = cast[ptr byte](cast[uint](currW) + 2); let w16 = mm_loadu_si128(pWeights)
        let lo8 = mm_and_si128(w16, mask); let hi8 = mm_and_si128(mm_srli_epi16(w16, 4), mask)
        mm256_store_ps(addr pd[i], mm256_mul_ps(mm256_sub_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(lo8)), eight), scalev))
        mm256_store_ps(addr pd[i+8], mm256_mul_ps(mm256_sub_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(mm_unpackhi_epi64(lo8, lo8))), eight), scalev))
        mm256_store_ps(addr pd[i+16], mm256_mul_ps(mm256_sub_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(hi8)), eight), scalev))
        mm256_store_ps(addr pd[i+24], mm256_mul_ps(mm256_sub_ps(mm256_cvtepi32_ps(mm256_cvtepi8_epi32(mm_unpackhi_epi64(hi8, hi8))), eight), scalev))
        currW = cast[ptr byte](cast[uint](currW) + 18); i += 32
  else:
    for i in 0..<size: that.setFloat(thatOffset + i, getFloat(thiz, thisOffset + i))

proc fillInPlace*(t: FloatTensor, offset, size: int, val: float32) {.inline.} =
  if t.kind == fkArray:
    for i in offset..<offset+size: t.dataPtr[i] = val
  else:
    for i in offset..<offset+size: setFloat(t, i, val)

proc addInPlace*(t, that: FloatTensor) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let size = t.size; let d1 = t.dataPtr; let d2 = that.dataPtr; var i = 0
    while i <= size - 8:
      mm256_store_ps(addr d1[i], mm256_add_ps(mm256_load_ps(addr d1[i]), mm256_load_ps(addr d2[i])))
      i += 8
    while i < size:
      d1[i] += d2[i]
      i += 1

proc multiplyInPlace*(t, that: FloatTensor) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let size = t.size; let d1 = t.dataPtr; let d2 = that.dataPtr; var i = 0
    while i <= size - 8:
      mm256_store_ps(addr d1[i], mm256_mul_ps(mm256_load_ps(addr d1[i]), mm256_load_ps(addr d2[i])))
      i += 8
    while i < size:
      d1[i] *= d2[i]
      i += 1

proc divideInPlace*(t: FloatTensor, offset, size: int, val: float32) {.inline.} =
  if t.kind == fkArray:
    let d = cast[ptr UncheckedArray[float32]](cast[uint](t.dataPtr) + offset.uint * 4)
    let v = mm256_set1_ps(val); var i = 0
    while i <= size - 8:
      mm256_store_ps(addr d[i], mm256_div_ps(mm256_load_ps(addr d[i]), v))
      i += 8
    while i < size:
      d[i] /= val
      i += 1

proc softmaxInPlace*(t: FloatTensor, offset, size: int) =
  if t.kind == fkArray:
    let d = cast[ptr UncheckedArray[float32]](cast[uint](t.dataPtr) + offset.uint * 4)
    var maxVal = d[0]
    for i in 1..<size:
      if d[i] > maxVal: maxVal = d[i]
    var i = 0; var sumv = mm256_setzero_ps(); let mv = mm256_set1_ps(maxVal)
    while i <= size - 8:
      let ev = exp_ps(mm256_sub_ps(mm256_load_ps(addr d[i]), mv))
      mm256_store_ps(addr d[i], ev)
      sumv = mm256_add_ps(sumv, ev)
      i += 8
    var sum = hsum(sumv)
    while i < size:
      let ev = exp(d[i] - maxVal)
      d[i] = ev; sum += ev; i += 1
    let vsum = mm256_set1_ps(sum)
    i = 0
    while i <= size - 8:
      mm256_store_ps(addr d[i], mm256_div_ps(mm256_load_ps(addr d[i]), vsum))
      i += 8
    while i < size:
      d[i] /= sum
      i += 1

proc saxpyInPlace*(t: FloatTensor, offset: int, that: FloatTensor, thatOffset: int, size: int, a: float32) {.inline.} =
  if t.kind == fkArray and that.kind == fkArray:
    let d1 = cast[ptr UncheckedArray[float32]](cast[uint](t.dataPtr) + offset.uint * 4)
    let d2 = cast[ptr UncheckedArray[float32]](cast[uint](that.dataPtr) + thatOffset.uint * 4)
    let av = mm256_set1_ps(a); var i = 0
    while i <= size - 8:
      mm256_store_ps(addr d1[i], mm256_fmadd_ps(av, mm256_load_ps(addr d2[i]), mm256_load_ps(addr d1[i])))
      i += 8
    while i < size:
      d1[i] += a * d2[i]
      i += 1

proc argmax*(t: FloatTensor, offset, size: int): int =
  var maxIndex = offset; var maxVal = getFloat(t, offset)
  for i in 1..<size:
    let v = getFloat(t, offset + i)
    if v > maxVal:
      maxVal = v
      maxIndex = offset + i
  return maxIndex

proc argmax*(t: FloatTensor): int = return argmax(t, 0, t.size)

proc rmsnorm*(outT, x: FloatTensor, weight: seq[float32], size: int, eps: float32) =
  if outT.kind == fkArray and x.kind == fkArray:
    let d_in = x.dataPtr; let pw = cast[ptr UncheckedArray[float32]](unsafeAddr weight[0])
    var sumv = mm256_setzero_ps(); var i = 0
    while i <= size - 8:
      let v = mm256_load_ps(addr d_in[i])
      sumv = mm256_fmadd_ps(v, v, sumv)
      i += 8
    var ss = hsum(sumv)
    while i < size:
      ss += d_in[i] * d_in[i]
      i += 1
    ss = 1.0f32 / sqrt(ss / size.float32 + eps)
    let d_out = outT.dataPtr; let sv = mm256_set1_ps(ss); i = 0
    while i <= size - 8:
      mm256_store_ps(addr d_out[i], mm256_mul_ps(mm256_load_ps(addr pw[i]), mm256_mul_ps(sv, mm256_load_ps(addr d_in[i]))))
      i += 8
    while i < size:
      d_out[i] = pw[i] * (ss * d_in[i])
      i += 1
