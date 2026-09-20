# MaiBot Gemini Transcribe Provider

这是一个针对 `gemini-3.5-transcribe` 的 MaiBot LLM Provider。小型聊天语音默认直接走
Interactions API 的 inline Base64；较大的音频才使用 Files API。Provider 生命周期内会复用
按 API key 隔离的 async Gemini Client。


## 1. 安装

把整个目录复制到：

```text
MaiBot/plugins/gemini35_transcribe/
```

安装依赖：

```bash
pip install -r plugins/gemini35_transcribe/requirements.txt
```

`google-genai >= 2.24.0` 才有本插件使用的 Interactions API async 入口。插件也默认安装
`pilk-nogil`，用于优先解码 QQ/Tencent SILK；如果专用解码器不可用或样本变体不兼容，仍会
回退到系统 FFmpeg。FFmpeg 仍用于 AMR、未知格式和专用解码失败的通用转码。

Interactions 的 inline 音频输入遵循 Google 当前接口：
`{"type": "audio", "data": "<base64>", "mime_type": "audio/..."}`。
`auto` 的默认 inline 上限是 256 KiB；超过上限的音频使用 Files API 并按配置删除上传文件。

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

音频传输与本地过滤：

```toml
[gemini]
# auto = 小音频 inline、大音频 Files；也可显式使用 inline 或 files。
audio_transport = "auto"
inline_max_bytes = 262144
# 只过滤 WAV/专用 SILK 解码后能高度确定为静音的录音。
local_speech_gate = true
```

已支持：

- `language_codes = ["zh-CN"]`
- `custom_vocabulary = ["MaiBot", "NapCat"]`
- `speaker_diarization = true`
- `word_timestamps = true`

注意：Google 不允许 `smart` 与说话人分离/词级时间戳同时使用，也不允许 `custom_vocabulary` 与说话人分离/词级时间戳同时使用；插件会提前报出明确错误。

### 音频与 fallback 行为

- `audio_transport = "auto"`：音频不超过 `inline_max_bytes` 时只发一次 Interactions 请求，不调用 Files API；大文件走 Files API。
- `audio_transport = "inline"`：强制 inline；Google 明确返回不支持/请求格式类错误时才回退 Files API。
- `audio_transport = "files"`：保留旧的 Files API 流程。
- dedicated transcribe 返回正常 HTTP 但空文本时不会重试上传；继续使用现有 Flash-Lite fallback。
- Files API 上传使用内存流，成功后默认删除远端临时文件。配置更新会平滑淘汰旧 Client，卸载时等待在途请求并关闭 async/sync Client。
- 本地 speech gate 只处理 PCM/WAV 的明显静音；它不是噪声检测器，响亮白噪声和不确定录音仍会提交 Gemini。

`pilk-nogil` 是 GPL-3.0 原生扩展，当前 PyPI 为 Windows/Linux 等常见 CPython 版本提供 wheel。
若部署平台无法安装该 wheel，仍可移除该可选包并依赖 FFmpeg fallback；需要验证真实 QQ/NapCat
SILK 样本（包括是否带 `0x02` Tencent 前缀）后再调整采样率或解码器选择。

## 5. 验证

重启/热重载插件后，发送一条 3~10 秒 QQ 语音。成功时 MaiBot 的 `voice` 任务会得到转写文本并按普通消息进入后续流程。

常见错误：

- `client_type not found`：插件未成功加载，检查 `_manifest.json` 与 `@LLMProvider` 是否都为 `gemini35.transcribe`。
- `interactions` 不存在：`google-genai` 太旧，执行 `pip install -U "google-genai>=2.24.0,<3"`。
- `未找到 Gemini API Key`：给 provider、插件 config 或 `GEMINI_API_KEY` 至少配置一个。
- `ffmpeg 无法转换`：上游给的是 AMR/未知格式，或 SILK 专用 decoder 未安装/无法识别；安装 `pilk-nogil`，或让 NapCat/Adapter 转成 WAV/OGG/MP3。
- `404/unsupported model`：确认模型名是 `gemini-3.5-transcribe`，不要填 `gemini-3.5-transcribe-live`。

## 6. 发布

在 GitHub Actions 中手动运行 `Release` workflow。它会从 `_manifest.json` 读取版本，校验
`CHANGELOG.md` 中存在同版本章节，检查 `v<version>` 尚未发布，然后创建带源码 zip 的 GitHub
Release。需要发布新版本时，先更新 `_manifest.json` 与 `CHANGELOG.md`，再运行 workflow；可选
选择 Draft 或 Prerelease。
