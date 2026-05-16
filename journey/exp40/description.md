## exp40 — exp35 + VGG Gram style loss

**Status: DONE 2026-05-15** (failed)

Gatys/Johnson style loss at fastai-default `style_weight=5000` dominates the
gradient signal (mean_loss = 0.037 vs typical 0.006 — 6× larger). Result:
~16–20% regression across **every** metric, faces especially. fastai's
5000 is tuned for pure neural style transfer with no flow loss; combined
with our MSE + LPIPS recipe it's way too aggressive. Would need
`style_weight ≈ 100–500` or an anneal schedule to be usable.

---
