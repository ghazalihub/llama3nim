import std/[parseopt, strutils, os, times, sequtils, sets]
import tensor, tokenizer, model_base, model, sampler

type
  Options = object
    modelPath: string
    prompt: string
    systemPrompt: string
    interactive: bool
    temperature: float32
    topp: float32
    seed: int64
    maxTokens: int
    stream: bool
    echo: bool

proc printUsage() =
  echo "Usage: llama3 [options]"
  echo ""
  echo "Options:"
  echo "  --model, -m <path>            required, path to .gguf file"
  echo "  --interactive, --chat, -i     run in chat mode"
  echo "  --instruct                    run in instruct (once) mode, default mode"
  echo "  --prompt, -p <string>         input prompt"
  echo "  --system-prompt, -sp <string> (optional) system prompt"
  echo "  --temperature, -temp <float>  temperature in [0,inf], default 0.1"
  echo "  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95"
  echo "  --seed <long>                 random seed, default cpuTime()"
  echo "  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default 512"
  echo "  --stream <boolean>            print tokens during generation, default true"
  echo "  --echo <boolean>              print ALL tokens to stderr, default false"

proc parseOptions(): Options =
  result.temperature = 0.1f32
  result.topp = 0.95f32
  result.seed = getTime().toUnix()
  result.maxTokens = 512
  result.stream = true
  result.echo = false

  var p = initOptParser()
  while true:
    p.next()
    case p.kind
    of cmdEnd: break
    of cmdShortOption, cmdLongOption:
      case p.key
      of "model", "m": result.modelPath = p.val
      of "prompt", "p": result.prompt = p.val
      of "system-prompt", "sp": result.systemPrompt = p.val
      of "interactive", "chat", "i": result.interactive = true
      of "instruct": result.interactive = false
      of "temperature", "temp": result.temperature = p.val.parseFloat().float32
      of "top-p": result.topp = p.val.parseFloat().float32
      of "seed", "s": result.seed = p.val.parseBiggestInt()
      of "max-tokens", "n": result.maxTokens = p.val.parseInt()
      of "stream": result.stream = p.val.parseBool()
      of "echo": result.echo = p.val.parseBool()
      of "help", "h":
        printUsage()
        quit(0)
      else:
        echo "Unknown option: ", p.key
        quit(1)
    of cmdArgument:
      discard

  if result.modelPath == "":
    echo "ERROR: --model <path> is required"
    printUsage()
    quit(1)
  if not result.interactive and result.prompt == "":
    echo "ERROR: --prompt is required in --instruct mode"
    printUsage()
    quit(1)

proc runInteractive(model: LlamaModel, sampler: Sampler, options: Options) =
  var state = newState(model.config)
  var conversationTokens = newSeq[int]()
  let cf = newChatFormat(model.tokenizer)
  conversationTokens.add(cf.beginOfText)
  if options.systemPrompt != "":
    conversationTokens.add(cf.encodeMessage(Message(role: roleSystem, content: options.systemPrompt)))

  var startPosition = 0
  while true:
    stdout.write("> ")
    stdout.flushFile()
    var userText: string
    try:
      userText = stdin.readLine()
    except EOFError:
      break
    if userText in ["quit", "exit"]: break

    conversationTokens.add(cf.encodeMessage(Message(role: roleUser, content: userText)))
    conversationTokens.add(cf.encodeHeader(Message(role: roleAssistant, content: "")))

    let responseTokens = generateTokens(model, state, startPosition, conversationTokens[startPosition..^1], cf.stopTokens, options.maxTokens, sampler, options.echo, proc(t: int) =
      if options.stream:
        if not model.tokenizer.isSpecialToken(t):
          stdout.write(model.tokenizer.decode(@[t]))
          stdout.flushFile()
    )

    conversationTokens.add(responseTokens)
    startPosition = conversationTokens.len

    var cleanResponse = responseTokens
    if cleanResponse.len > 0 and cf.stopTokens.contains(cleanResponse[^1]):
      cleanResponse.setLen(cleanResponse.len - 1)

    if not options.stream:
      echo model.tokenizer.decode(cleanResponse)

    if responseTokens.len > 0 and not cf.stopTokens.contains(responseTokens[^1]):
      echo "Ran out of context length..."
      break

proc runInstructOnce(model: LlamaModel, sampler: Sampler, options: Options) =
  let state = newState(model.config)
  let cf = newChatFormat(model.tokenizer)
  var promptTokens = newSeq[int]()
  promptTokens.add(cf.beginOfText)
  if options.systemPrompt != "":
    promptTokens.add(cf.encodeMessage(Message(role: roleSystem, content: options.systemPrompt)))
  promptTokens.add(cf.encodeMessage(Message(role: roleUser, content: options.prompt)))
  promptTokens.add(cf.encodeHeader(Message(role: roleAssistant, content: "")))

  let responseTokens = generateTokens(model, state, 0, promptTokens, cf.stopTokens, options.maxTokens, sampler, options.echo, proc(t: int) =
    if options.stream:
      if not model.tokenizer.isSpecialToken(t):
        stdout.write(model.tokenizer.decode(@[t]))
        stdout.flushFile()
  )

  var cleanResponse = responseTokens
  if cleanResponse.len > 0 and cf.stopTokens.contains(cleanResponse[^1]):
    cleanResponse.setLen(cleanResponse.len - 1)

  if not options.stream:
    echo model.tokenizer.decode(cleanResponse)
  else:
    echo ""

proc main() =
  let options = parseOptions()
  let model = loadModel(options.modelPath, options.maxTokens)
  let sampler = selectSampler(model.config.vocabularySize, options.temperature, options.topp, options.seed)

  if options.interactive:
    runInteractive(model, sampler, options)
  else:
    runInstructOnce(model, sampler, options)

if isMainModule:
  main()
