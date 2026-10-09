[English](README.md) · **简体中文**

# 小输入，直接跑

[返回 mJev](../README.zh-CN.md) · [HF 环境准备](../docs/models.zh-CN.md)

通用图片与视频示例使用项目自制媒体；文档示例使用保留 CC-BY-2.0 署名的真实验证图片。将 `MODEL_DIR` 指向已下载模型后，选一个输入即可：

| 示例 | 输入 | 问题数 |
| --- | --- | ---: |
| 一张图片，一道题 | [single.json](single.json) | 1 |
| 一张图片，多道题 | [multiple.json](multiple.json) | 2 |
| 静态视频：颜色、形状与背景 | [video-demo/input.json](video-demo/input.json) | 3 |
| 动画：动作、对象与事件顺序 | [motion-demo/input.json](motion-demo/input.json) | 3 |
| 研究表格：四对中英文问题 | [docjev/multiple.json](docjev/multiple.json) | 8 条语言记录 |

```bash
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json --check-only
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-demo.json
```

命令在仓库根目录执行，媒体路径相对于输入 JSON 解析。视频示例需要系统 FFmpeg 与[环境指南](../docs/models.zh-CN.md)中的 HF 视频依赖。

## 已有输出记录

[静态视频结果](video-demo/recorded-output.json)与[动画结果](motion-demo/recorded-output.json)保留了 **Qwen/Qwen3-VL-4B-Instruct** 的实际 HF 输出，模型 revision 为 `ebb281ec70b05090aa6165b016eac8ec08e71b17`。使用所选模型运行输入，即可获得对应结果。

这些合成示例用于检查运行流程。准确率评测应使用保留原始标签的数据集；候选概率只在本题选项集合内比较。

## mJev-Doc 文档示例

[真实验证表格与中英文对应题](docjev/README.md)包含四对问题，候选数量各为 4、3、3、2，保留原始候选顺序与参考答案。文档图片保留 CC-BY-2.0 许可及署名。

```bash
python demo_docjev.py --model "$MODEL_DIR" \
  --input examples/docjev/multiple.json --output outputs/document-demo.json
```

`MODEL_DIR` 可指向官方 Qwen3-VL 基础模型或完整 mJev-Doc checkpoint。示例的已有输出来自 mJev-Doc；完整模型权重另行下载。
