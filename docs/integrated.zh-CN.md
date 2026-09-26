[English](integrated.md) · **简体中文**

# 集成实验 runtime

> 最新验证状态与后端边界统一见 [验证入口](validation_current.zh-CN.md)。本页历史结果保留原记录；未记录的模型 revision、完整依赖或运行配置标为未核实，不从当前默认值推断。

组合 main `ca755f8` 参考包/评估器与 mJev `59b38ea` 服务/Tree-KV runtime，共用确定性决策和稳定的候选 softmax。`mjev/` dense-SDPA 参考路径仍保留。服务默认 causal，带 mask 的 Tree-KV 为实验模式；runtime 与参考 hooks 不得同时启用。

该 runtime 仍调用 vLLM `generate` 传递单步分数，不是文本解码循环，但**不满足完全禁止调用 generate 的要求**。没有新增模型 head 或训练。

CPU selected projection 对照的是相同 BF16 权重的完整 FP32 投影，不是完整 30B/TP 一致性证明。全模型 Tree-KV 与 dense mask 的一致性仍待验证；不同 prompt/后端不可视作精确 logits 对照。

## CPU 验证

```bash
docker run --rm -w /workspace -e PYTHONPATH=/workspace/runtime:/workspace \
  -v "$PWD:/workspace:ro" --entrypoint python3 mjev-av-pilot:0.1 \
  -m pytest -q -p no:cacheprovider tests/runtime
```

## GPU 部署

仅在四张 GPU 可用后执行：

```bash
export MODEL_DIR=/path/to/Qwen3-Omni-30B-A3B-Instruct
export DATA_DIR=/path/to/mjev_multiquestion_le4000_v1
bash scripts/serve_integrated.sh
curl -f http://127.0.0.1:17005/health
```

需已有 `mjev-av-pilot:0.1` 镜像，否则在仓库根目录运行 `docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .`。启动脚本创建独立容器，不停止已有服务，API 仅绑定 loopback。

## 评测

在参考 CPU 镜像中执行，代码位于 PYTHONPATH 且数据可读：

```bash
python3 benchmarks/integrated/evaluate_service.py --root /data --out /results/prepared --prepare-only
python3 benchmarks/integrated/evaluate_service.py --root /data --out /results/smoke --limit-per-modality 2
python3 benchmarks/integrated/evaluate_service.py --root /data --out /results/full
```

输出必须为新目录；错误立即停止并保存 `ERROR.json`，不重试、不静默遗漏。运行器仅发送媒体、问题、选项和 ID。374 道原生/分歧 Clotho 问题排除于 MCQ 指标；226 道原始 yes/no 一致参考题使用另行记录的适配器。带音轨视频明确启用官方 audio-in-video。

先前不超过 4000 tokens 的长度只适用于原 processor/template，不保证新 runtime 长度。不会裁剪；超限或解码错误应调查而非静默跳过。Accuracy/NLL/Brier/ECE 是逐题指标，不是官方榜单分数。延迟是客户端每组请求墙钟时间，不是每题时间。两模式交替执行，不能认定为经过验证的冷缓存计时。

HTTP 评估器现在记录客户端源码和依赖，但无法证明服务端模型 revision、numerics 或依赖；这些字段明确为 unverified_not_attested，不能当作完整服务端复现证据。

## 媒体访问与启动检查

实验 HTTP 服务在调用 vLLM 之前执行自身的媒体边界检查：

- 本地路径和 `file:` URL 必须位于 `MJEV_ALLOWED_MEDIA_ROOTS` 内（默认 `/data`；多个目录用系统路径分隔符连接）。只读取普通文件，拒绝通过符号链接越界。启动脚本将 `/data` 只读挂载。
- 默认禁止远程读取。可用 `MJEV_ALLOWED_MEDIA_HOSTS` 指定逗号分隔的精确主机名；仅允许 HTTPS 443，拒绝 URL 凭据、非公网地址和重定向。连接固定到已校验的 IP，并按主机名验证 TLS。
- `MJEV_MAX_MEDIA_BYTES` 默认每项 536870912 字节。读取／解码前检查文件大小和 data URL 编码长度，并使用有界读取。传给 vLLM 的是已验证的内容，不再重新打开路径或下载。

启动要求 vLLM 0.25.1 和全部 runtime hooks 安装成功；失败时终止进程，包含新启动的 Python worker。`/v1/mjev/health` 检查前端 hooks，同时应使用 vLLM 标准 `/health` 检查引擎健康状态。这些检查不能替代全模型 Tree-KV 数值一致性验证。
