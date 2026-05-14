---
name: Bug report
about: Something is broken
title: "[BUG] "
labels: bug
assignees: ShadowGodd1
---

## Describe the bug

A clear description of what the bug is.

## To Reproduce

Minimal reproducible example:

```python
import pandas as pd
from ds_toolkit.core import DataProfiler

df = pd.DataFrame({"a": [1, 2, None]})
# ... what you ran
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full traceback:

```
Traceback (most recent call last):
  ...
```

## Environment

- ds-toolkit version: <!-- python -c "import ds_toolkit; print(ds_toolkit.__version__)" -->
- Python version:
- OS:
- sklearn version:
- pandas version:

## Additional context

Any other context about the problem.
