# Little Fig — Research Plan

## Status of Each Training Tier

### Tier 1: LoRA (Streaming) — ✅ SOLID
- Converges correctly (tested head-to-head vs FP32 PEFT LoRA)
- No changes needed. Works as intended.

### Tier 2: LISA — ⚠️ ONE ISSUE: Random layer selection
- The mechanism works (tested, passes all tests)
- BUT: layers are selected UNIFORMLY at random
- Observation showed loss sensitivity varies 100x across layers (some shift loss by 6.3, others by 0.03)
- **Research opportunity: Direction 3 (loss-sensitivity-guided selection)**

### Tier 3: MeZO — ✅ IMPROVED (FigMeZO shipped)
- Original MeZO: gradient estimate cosine = ±0.0008 (random)
- FigMeZO with inverse shaping: −18.6% loss improvement (validated, 3 seeds)
- **DONE. Pushed to git.**

### Tier 4: LOMO — ⚠️ TWO ISSUES
- Uses pure SGD (no momentum) — this is by design (O(1) memory)
- BUT: all layers get the same learning rate
- Same loss sensitivity issue as LISA — layers with different curvatures shouldn't get same LR
- **Research opportunity: per-layer LR scaling based on measured sensitivity**

### FigQuant — ⚠️ ONE INEFFICIENCY
- k-means runs per-layer (8 iterations per layer × 50 layers = 400 iterations)
- Observation showed all codebooks are essentially the same (L2 distance = 0.019 between layers, drift from NF4 = 0.1 for ALL layers equally)
- **Research opportunity: Direction 1 (shared codebook)**

---

## Research Direction 1: Shared Codebook

**Observation:** All 50 GPT-2 codebooks land within 0.019 L2 of each other.
Early layers drift 0.1007 from NF4. Late layers drift 0.1049. Essentially identical.

**Hypothesis:** A single codebook computed from the full model's weight distribution
should match or beat per-layer codebooks at 50x less computation.

**Experiment:**
1. Concatenate ALL weight matrices, run k-means once → "global codebook"
2. Quantize each layer using the global codebook (skip per-layer k-means)
3. Compare MSE vs per-layer codebooks on TinyLlama

**Success metric:** MSE within 5% of per-layer, at 50x less quantization time.

---

## Research Direction 3: Loss-Sensitivity-Guided Layer Selection

**Observation:** Layer 0 c_attn shifts loss by −6.3 at perturbation 0.1.
Layer 11 c_proj shifts loss by −0.03. 200x difference.

**Hypothesis:** LISA should unfreeze high-sensitivity layers more often.
LOMO should use smaller LR on high-sensitivity layers (sharp curvature = need small steps).

**Experiment:**
1. Run one "probe" pass: for each layer, compute |ΔL| at scale 0.01
2. Use |ΔL| as LISA sampling weight (replace uniform random)
3. Compare convergence: random LISA vs sensitivity-weighted LISA, 100 steps

**Success metric:** Lower final loss in same number of steps.

---

## Research Direction 4: Ember Memory Training Optimization

**The integration exists** (ember_integration.py generates training data).
**The question:** Can we make the memory tokens train faster/better?

**Observation from Self-Memory System paper (Conway 2005):**
Human memory retrieves based on CURRENT GOALS, not similarity.
The current Ember recall uses semantic similarity (vector search).
Training the model to emit <|mem_recall|> based on goal-relevance
(not just topic-similarity) would be closer to human cognition.

**Observation from Platonic Representation:**
All models converge to the same representation geometry at scale.
This means the EMBEDDING SPACE for Ember's vector index is model-agnostic.
We could pre-compute a "universal memory embedding" that works across models.

**Experiment:**
1. Generate goal-conditioned recall training data (not just topic-match)
2. Compare: topic-similarity recall vs goal-conditioned recall
3. Measure: does the model learn to retrieve memories relevant to
   the current TASK rather than the current TOPIC?

---

## Research Direction 5: RLM-Style Recursive Memory for Long-Context

**From RLM paper:** Store document outside context, let model write code to navigate.
**From Ember:** Memory is structured, immutable, queryable.

**Combination (original):** Instead of storing raw text in a Python variable,
store it in an Ember database. The model writes EMBER QUERIES instead of
regex/slice operations. This gives it:
- Graph traversal (follow connections between memories)
- Timeline queries (what happened before/after X)
- Confidence-aware retrieval (only trust high-confidence memories)
- Episode segmentation (natural chunking boundaries)

**This is not RLM + Ember bolted together.** It's a different retrieval paradigm:
RLM uses regex/slice (syntax). Ember uses semantics + structure + time.

**Experiment:** Compare on a long-context task:
- Baseline: raw text in Python variable (RLM-style)
- Ours: text stored in Ember, model queries via MemoryProtocol
- Measure: accuracy on multi-hop questions over long documents

---

## Execution Order

1. ✅ FigMeZO (done, −18.6%)
2. → Direction 1: Shared Codebook (fastest to test, clear hypothesis)
3. → Direction 3: Sensitivity-guided LISA/LOMO (needs probe step)
4. → Direction 4: Goal-conditioned Ember recall training
5. → Direction 5: RLM × Ember (largest scope, needs working model first)

---

## Rules
- NO ideas from other papers used directly
- Every claim backed by OUR OWN experiment
- If it doesn't beat baseline in controlled test, it doesn't ship
- Write findings BEFORE implementing changes
