# Journal

Module-by-module notes capturing what I learned, where I got stuck, and the moments where things clicked, while building SkyAI.

## Why this exists

The code in this repo is one artifact of the learning process. The journal is the other. It's meant to be read alongside the code, in module order, to follow the reasoning that shaped each design choice.

If you came here to evaluate this project, start with the journal. The code shows what I built; the journal shows what I understood.

## Run results

[`runs/`](./runs/) holds one document per completed training run - the results, the exact
protocol they were measured under, how to state the finding defensibly, and the caveats
that must stay attached to the numbers. These are the empirical counterpart to the module
notes: the entries below record what I understood, `runs/` records what actually happened
when it was trained.

- [`runs/gpt2-muon-124m.md`](./runs/gpt2-muon-124m.md) - faithful GPT-2 with the Muon recipe
- [`runs/skyai-124m.md`](./runs/skyai-124m.md) - the modern stack, head-to-head against it

## Structure

Entries prefixed `prereq-` cover the foundations I built up *before* SkyAI proper -
Karpathy's Zero to Hero series and 3Blue1Brown's neural-network series, worked through in
order and journaled as I went. These are deliberately short: capture-the-reaction notes,
not polished essays.

```
prereq-3b1b.md              neural nets / backprop / attention, the visual intuition
prereq-micrograd.md         autograd from scratch
makemore-bigrams.md         counting, then the same thing as a neural net
makemore-mlp.md             Bengio-style MLP language model
makemore-activations.md     init, BatchNorm, and why activations die
makemore-backprop.md        backprop ninja - every gradient by hand
makemore-wavenet.md         hierarchical / dilated architecture
lets-build-gpt.md           attention and the transformer block
gpt-tokenizer.md            BPE from scratch, gpt2 vs cl100k
01-reproduce-gpt2.md        reproducing GPT-2 124M in this repo
```

## What's in each entry

Loose template, not strict:

- **What I'm building** - the module under construction
- **Concepts I had to internalize** - the math/intuition that took real work
- **What surprised me** - bugs, unintuitive behavior, "wait, why does this work?"
- **What I'd do differently** - design choices I might revisit
- **Open questions** - things to come back to later
