**English** · [简体中文](README.zh-CN.md)

# Small inputs, ready to run

[Back to mJev](../README.md) · [HF setup](../docs/models.md)

All bundled media is project-created. Pick an input after setting `MODEL_DIR` to your downloaded checkpoint:

| Example | Input | Questions |
| --- | --- | ---: |
| One image, one question | [single.json](single.json) | 1 |
| One image, multiple questions | [multiple.json](multiple.json) | 2 |
| Static video: color, shape and background | [video-demo/input.json](video-demo/input.json) | 3 |
| Animation: motion, object and event order | [motion-demo/input.json](motion-demo/input.json) | 3 |

```bash
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json --check-only
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-demo.json
```

Commands run from the repository root. Media paths resolve relative to their input JSON. Video examples require system FFmpeg and the HF video dependencies in the [setup guide](../docs/models.md).

## Recorded outputs

[Static-video output](video-demo/recorded-output.json) and [animation output](motion-demo/recorded-output.json) retain actual HF predictions from **Qwen/Qwen3-VL-4B-Instruct**, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Run the inputs to obtain results for your chosen checkpoint.

These synthetic fixtures exercise the workflow. Use original labeled datasets for accuracy evaluation; candidate probabilities are relative to each question's choices.

## DocJev documents

[Real validation table with paired Chinese/English questions](docjev/README.md). Run `python demo_docjev.py --model /path/to/checkpoint --input examples/docjev/multiple.json`. The image retains its CC-BY-2.0 attribution.
