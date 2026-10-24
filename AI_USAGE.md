# How AI assistants were used in this project

Interviewers ask. This is the honest answer.

**Used for**
- Scaffolding the repo layout, Makefile, CI workflow and the first version of each module.
- Drafting ~40 of the 100 eval questions (marked `"author": "llm"`). Every one was checked by hand against the real label text before being committed.
- Explaining library APIs (pgvector, sentence-transformers) while I was learning them.

**Not used for**
- The numbers in the README. Every metric comes from `make eval` on this machine.
- The hand-written eval questions (`"author": "sreya"`).
- Design decisions in the README. Those are mine and I can defend each one.

**How I verified AI output**
- Every module has tests I wrote or read line by line (`make test`).
- I rewrote anything I could not explain. If a line is in this repo, I can walk through it.
