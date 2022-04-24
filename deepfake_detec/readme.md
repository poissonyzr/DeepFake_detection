# Deepfake Detection Operator (Pytorch)

Authors: poissonyzr

## Overview


## Interface

```python
__call__(self, audio_path: str)
```

**Args:**

- audio_path:
  - the input audio path
  - supported types: str

**Returns:**

The Operator returns a list['name',score] containing possibility of fake videos.


## Requirements

You can get the required python package by [requirements.txt](./requirements.txt).

## How it works

Trained with cutmix SE-ResNeXT 50 on frame-level detected faces

## Reference

[1]. https://github.com/jphdotam/DFDC