import std/[streams, tables, strutils]

type
  GGMLType* = enum
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q4_2 = 4, Q4_3 = 5, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11,
    Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    I8 = 16, I16 = 17, I32 = 18

  MetadataValueType* = enum
    mvtUint8 = 0, mvtInt8 = 1, mvtUint16 = 2, mvtInt16 = 3,
    mvtUint32 = 4, mvtInt32 = 5, mvtFloat32 = 6, mvtBool = 7,
    mvtString = 8, mvtArray = 9, mvtUint64 = 10, mvtInt64 = 11,
    mvtFloat64 = 12

  MetadataValue* = object
    case kind*: MetadataValueType
    of mvtUint8: u8*: uint8
    of mvtInt8: i8*: int8
    of mvtUint16: u16*: uint16
    of mvtInt16: i16*: int16
    of mvtUint32: u32*: uint32
    of mvtInt32: i32*: int32
    of mvtFloat32: f32*: float32
    of mvtBool: b*: bool
    of mvtString: s*: string
    of mvtArray: arr*: seq[MetadataValue]
    of mvtUint64: u64*: uint64
    of mvtInt64: i64*: int64
    of mvtFloat64: f64*: float64

  GGUFTensorInfo* = object
    name*: string
    dimensions*: seq[int64]
    ggmlType*: GGMLType
    offset*: int64

  GGUF* = ref object
    magic*: uint32
    version*: uint32
    tensorCount*: uint64
    metadataKvCount*: uint64
    metadata*: Table[string, MetadataValue]
    tensorInfos*: Table[string, GGUFTensorInfo]
    tensorDataOffset*: int64

proc readGGUFString(s: Stream): string =
  let len = s.readUint64()
  if len == 0: return ""
  result = s.readStr(len.int)

proc readMetadataValue(s: Stream): MetadataValue

proc readMetadataValueOfType(s: Stream, vType: MetadataValueType): MetadataValue =
  case vType:
  of mvtUint8: result = MetadataValue(kind: mvtUint8, u8: s.readUint8())
  of mvtInt8: result = MetadataValue(kind: mvtInt8, i8: s.readInt8())
  of mvtUint16: result = MetadataValue(kind: mvtUint16, u16: s.readUint16())
  of mvtInt16: result = MetadataValue(kind: mvtInt16, i16: s.readInt16())
  of mvtUint32: result = MetadataValue(kind: mvtUint32, u32: s.readUint32())
  of mvtInt32: result = MetadataValue(kind: mvtInt32, i32: s.readInt32())
  of mvtFloat32: result = MetadataValue(kind: mvtFloat32, f32: s.readFloat32())
  of mvtBool: result = MetadataValue(kind: mvtBool, b: s.readUint8() != 0)
  of mvtString: result = MetadataValue(kind: mvtString, s: s.readGGUFString())
  of mvtUint64: result = MetadataValue(kind: mvtUint64, u64: s.readUint64())
  of mvtInt64: result = MetadataValue(kind: mvtInt64, i64: s.readInt64())
  of mvtFloat64: result = MetadataValue(kind: mvtFloat64, f64: s.readFloat64())
  of mvtArray:
    let itemType = s.readUint32().MetadataValueType
    let len = s.readUint64()
    var arr = newSeq[MetadataValue](len)
    for i in 0..<len.int:
      arr[i] = s.readMetadataValueOfType(itemType)
    result = MetadataValue(kind: mvtArray, arr: arr)

proc readMetadataValue(s: Stream): MetadataValue =
  let vTypeID = s.readUint32()
  let vType = vTypeID.MetadataValueType
  return s.readMetadataValueOfType(vType)

proc loadGGUF*(filename: string): GGUF =
  let s = newFileStream(filename, fmRead)
  if s == nil: raise newException(IOError, "Cannot open file " & filename)

  result = GGUF()
  result.magic = s.readUint32()
  if result.magic != 0x46554747: # "GGUF"
    s.close()
    raise newException(ValueError, "Invalid GGUF magic")

  result.version = s.readUint32()
  if result.version != 2 and result.version != 3:
    s.close()
    raise newException(ValueError, "Unsupported GGUF version: " & $result.version)

  result.tensorCount = s.readUint64()
  result.metadataKvCount = s.readUint64()

  result.metadata = initTable[string, MetadataValue]()
  for i in 0..<result.metadataKvCount.int:
    let key = s.readGGUFString()
    let val = s.readMetadataValue()
    result.metadata[key] = val

  result.tensorInfos = initTable[string, GGUFTensorInfo]()
  for i in 0..<result.tensorCount.int:
    var ti: GGUFTensorInfo
    ti.name = s.readGGUFString()
    let n_dims = s.readUint32()
    ti.dimensions = newSeq[int64](n_dims)
    for j in 0..<n_dims.int:
      ti.dimensions[j] = s.readInt64()
    let ggmlTypeID = s.readUint32()
    ti.ggmlType = ggmlTypeID.GGMLType
    ti.offset = s.readInt64()
    result.tensorInfos[ti.name] = ti

  # Alignment
  var alignment: int64 = 32
  if result.metadata.contains("general.alignment"):
    let alVal = result.metadata["general.alignment"]
    if alVal.kind == mvtUint32: alignment = alVal.u32.int64
    elif alVal.kind == mvtInt32: alignment = alVal.i32.int64

  let currentPos = s.getPosition()
  let padding = (alignment - (currentPos mod alignment)) mod alignment
  result.tensorDataOffset = currentPos + padding
  s.close()

proc getAlignment*(gguf: GGUF): int =
  if gguf.metadata.contains("general.alignment"):
    let alVal = gguf.metadata["general.alignment"]
    if alVal.kind == mvtUint32: return alVal.u32.int
    elif alVal.kind == mvtInt32: return alVal.i32.int
  return 32
