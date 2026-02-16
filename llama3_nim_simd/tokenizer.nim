import std/[tables, strutils, re, unicode, sequtils, algorithm, sets]

type
  Vocabulary* = ref object
    tokens*: seq[string]
    tokenToIndex*: Table[string, int]

  Tokenizer* = ref object
    vocab*: Vocabulary
    merges*: Table[(int, int), int]
    pattern*: Regex
    specialTokens*: Table[string, int]
    indexToSpecialToken*: Table[int, string]

  MessageRole* = enum
    roleSystem = "system", roleUser = "user", roleAssistant = "assistant"

  Message* = object
    role*: MessageRole
    content*: string

  ChatFormat* = ref object
    tokenizer*: Tokenizer
    beginOfText*, startHeader*, endHeader*, endOfTurn*, endOfText*, endOfMessage*: int
    stopTokens*: HashSet[int]

proc newVocabulary*(tokens: seq[string]): Vocabulary =
  result = Vocabulary(tokens: tokens, tokenToIndex: initTable[string, int]())
  for i, t in tokens:
    result.tokenToIndex[t] = i

proc bytesToUnicode*(): Table[byte, int] =
  result = initTable[byte, int]()
  var bs = newSeq[int]()
  for i in 0x21..0x7E: bs.add(i)
  for i in 0xA1..0xAC: bs.add(i)
  for i in 0xAE..0xFF: bs.add(i)

  var cs = bs.mapIt(it)
  var n = 0
  for b in 0..255:
    if not bs.contains(b):
      bs.add(b)
      cs.add(256 + n)
      n += 1

  for i in 0..<bs.len:
    result[bs[i].byte] = cs[i]

let byteEncoder* = bytesToUnicode()
var byteDecoder* = initTable[int, byte]()
for k, v in byteEncoder: byteDecoder[v] = k

proc newTokenizer*(vocab: Vocabulary, merges: seq[(int, int)], patternStr: string, specialTokens: Table[string, int]): Tokenizer =
  result = Tokenizer(
    vocab: vocab,
    merges: initTable[(int, int), int](),
    pattern: re("(*UCP)" & patternStr),
    specialTokens: specialTokens,
    indexToSpecialToken: initTable[int, string]()
  )
  for pair in merges:
    let (f, s) = pair
    if f >= vocab.tokens.len or s >= vocab.tokens.len: continue
    let combined = vocab.tokens[f] & vocab.tokens[s]
    if vocab.tokenToIndex.contains(combined):
      result.merges[pair] = vocab.tokenToIndex[combined]

  for k, v in specialTokens:
    result.indexToSpecialToken[v] = k

proc isSpecialToken*(t: Tokenizer, tokenIndex: int): bool =
  return t.indexToSpecialToken.contains(tokenIndex)

proc getStats(ids: seq[int]): Table[(int, int), int] =
  result = initTable[(int, int), int]()
  for i in 0..<ids.len - 1:
    let pair = (ids[i], ids[i+1])
    result[pair] = result.getOrDefault(pair, 0) + 1

proc merge(ids: seq[int], pair: (int, int), idx: int): seq[int] =
  result = newSeq[int]()
  var i = 0
  while i < ids.len:
    if i < ids.len - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      result.add(idx)
      i += 2
    else:
      result.add(ids[i])
      i += 1

proc encodeChunk(t: Tokenizer, mappedChunk: string): seq[int] =
  var ids = newSeq[int]()
  for r in mappedChunk.toRunes():
    let s = $r
    if t.vocab.tokenToIndex.contains(s):
      ids.add(t.vocab.tokenToIndex[s])

  while ids.len >= 2:
    let stats = getStats(ids)
    var bestPair = (-1, -1)
    var minMergeIdx = int.high

    for pair in stats.keys:
      if t.merges.contains(pair):
        let mergeIdx = t.merges[pair]
        if mergeIdx < minMergeIdx:
          minMergeIdx = mergeIdx
          bestPair = pair

    if bestPair == (-1, -1): break
    ids = merge(ids, bestPair, minMergeIdx)
  return ids

proc encodeOrdinary*(t: Tokenizer, text: string): seq[int] =
  result = newSeq[int]()
  for chunk in text.findAll(t.pattern):
    var mappedChunk = ""
    for b in chunk:
      mappedChunk.add(unicode.toUTF8(Rune(byteEncoder[b.byte])))
    result.add(t.encodeChunk(mappedChunk))

proc encode*(t: Tokenizer, text: string): seq[int] =
  return t.encodeOrdinary(text)

proc decode*(t: Tokenizer, tokens: seq[int]): string =
  var s = ""
  for token in tokens:
    if token >= 0 and token < t.vocab.tokens.len:
      s.add(t.vocab.tokens[token])

  var bytes = newSeq[byte]()
  for r in s.toRunes():
    if byteDecoder.contains(r.int):
      bytes.add(byteDecoder[r.int])

  result = ""
  for b in bytes: result.add(char(b))

proc newChatFormat*(tokenizer: Tokenizer): ChatFormat =
  let st = tokenizer.specialTokens
  result = ChatFormat(
    tokenizer: tokenizer,
    beginOfText: st.getOrDefault("<|begin_of_text|>", -1),
    startHeader: st.getOrDefault("<|start_header_id|>", -1),
    endHeader: st.getOrDefault("<|end_header_id|>", -1),
    endOfTurn: st.getOrDefault("<|eot_id|>", -1),
    endOfText: st.getOrDefault("<|end_of_text|>", -1),
    endOfMessage: st.getOrDefault("<|eom_id|>", -1),
    stopTokens: initHashSet[int]()
  )
  if result.endOfText != -1: result.stopTokens.incl(result.endOfText)
  if result.endOfTurn != -1: result.stopTokens.incl(result.endOfTurn)

proc encodeHeader*(cf: ChatFormat, message: Message): seq[int] =
  result = newSeq[int]()
  if cf.startHeader != -1: result.add(cf.startHeader)
  result.add(cf.tokenizer.encodeOrdinary($message.role))
  if cf.endHeader != -1: result.add(cf.endHeader)
  result.add(cf.tokenizer.encodeOrdinary("\n"))

proc encodeMessage*(cf: ChatFormat, message: Message): seq[int] =
  result = cf.encodeHeader(message)
  result.add(cf.tokenizer.encodeOrdinary(strutils.strip(message.content)))
  if cf.endOfTurn != -1: result.add(cf.endOfTurn)

proc replaceControlCharacters*(s: string): string =
  result = ""
  for r in s.toRunes():
    if r.int < 32 and r.int != 10:
      result.add("\\u" & r.int.toHex(4))
    else:
      result.add(r.toUTF8())
