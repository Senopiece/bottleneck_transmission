## Asyncronus deletion (+corruption) resilient rateless transmission

This repo consists of:
- [impls](src/impls) - competitor protocol implementaions (our proposed ideas as well as sota solutions like lt and raptor codes)
- [docs](src/docs) - documentation to related implementations
   > NOTE: suggest to start from `_interface.md`
- [main.ipynb](src/main.ipynb) - evaluation and comparison of the methods

---
NOTE: at first it was an attempt to transfer more data over tight channels - thats why its named bottlebeck. But appears that it is also little more optimal than lt codes even within the domain of the lt codes on small deletion probability (TODO: test on raptor codes). Still all the chain implementations are very computationally expensive so they become unfeasible on large packet sizes.