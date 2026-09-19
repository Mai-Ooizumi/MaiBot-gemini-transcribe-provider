# MaiBot Gemini Transcribe Provider

这是一个针对 `gemini-3.5-transcribe` 的 MaiBot LLM Provider。


## 1. 安装

把整个目录复制到：

```text
MaiBot/plugins/gemini35_transcribe/
```

安装依赖：

```bash
pip install -r plugins/gemini35_transcribe/requirements.txt
```

`google-genai >= 2.3.0` 才有 Interactions API。

如果 QQ/NapCat/SnowLuma 给出的语音不是 Gemini 可直接接受的格式，插件会尝试调用系统 `ffmpeg` 转成 16 kHz 单声道 WAV。建议系统已安装 ffmpeg。Tencent SILK 是否能直接转码取决于你的 ffmpeg 构建；最稳妥的是让 Adapter 先输出 WAV/OGG/MP3。

## 2. model_config.toml

在现有配置结构中新增一个 Provider 和一个 Model（不要把 `[[api_providers]]` 改成别的 TOML 结构）：

```toml
[[api_providers]]
name = "google-gemini35-transcribe"
base_url = "https://generativelanguage.googleapis.com"
api_key = "你的_GEMINI_API_KEY"
client_type = "gemini35.transcribe"
max_retry = 3
retry_interval = 5
timeout = 120

[[models]]
name = "gemini-3.5-transcribe"
model_identifier = "gemini-3.5-transcribe"
api_provider = "google-gemini35-transcribe"
price_in = 0
price_out = 0
visual = false
force_stream_mode = false

[model_task_config.voice]
model_list = ["gemini-3.5-transcribe"]
max_tokens = 4096
hard_timeout = 180.0
selection_strategy = "sequential"
```

MaiBot 当前 `PluginLLMClient` 会把完整 `api_provider` 快照（包括鉴权配置）传给插件，所以通常只需要在 `model_config.toml` 配一次 Key。若你希望把 Key 与模型配置分开，也可以把它写入本插件的 `config.toml`：

```toml
[gemini]
api_key = "你的_GEMINI_API_KEY"
```

也可设置环境变量 `GEMINI_API_KEY`。

## 3. bot_config.toml

```toml
[voice]
enable_asr = true
```

## 4. 可选转写设置

`config.toml` 默认 `mode = "verbatim"`。若更希望把聊天语音整理得可读，可以改为：

```toml
[gemini]
mode = "smart"
```

已支持：

- `language_codes = ["zh-CN"]`
- `custom_vocabulary = ["MaiBot", "NapCat"]`
- `speaker_diarization = true`
- `word_timestamps = true`

注意：Google 不允许 `smart` 与说话人分离/词级时间戳同时使用，也不允许 `custom_vocabulary` 与说话人分离/词级时间戳同时使用；插件会提前报出明确错误。

## 5. 验证

重启/热重载插件后，发送一条 3~10 秒 QQ 语音。成功时 MaiBot 的 `voice` 任务会得到转写文本并按普通消息进入后续流程。

常见错误：

- `client_type not found`：插件未成功加载，检查 `_manifest.json` 与 `@LLMProvider` 是否都为 `gemini35.transcribe`。
- `interactions` 不存在：`google-genai` 太旧，执行 `pip install -U "google-genai>=2.3.0,<3"`。
- `未找到 Gemini API Key`：给 provider、插件 config 或 `GEMINI_API_KEY` 至少配置一个。
- `ffmpeg 无法转换`：上游给的是 SILK/AMR 等格式；让 NapCat/Adapter 转成 WAV/OGG/MP3 最稳妥。
- `404/unsupported model`：确认模型名是 `gemini-3.5-transcribe`，不要填 `gemini-3.5-transcribe-live`。
