# Coordinate and NeCo-Style Ranking Consistency — Design

Date: 2026-11-15
Status: Implemented in the uncommitted working tree, with bounded verification and partial acceptance as of 2026-11-17. The implementation and subsequent repair audits are recorded below; long numerical streams, accelerator memory/production cost, exact full-trainer replay and scientific efficacy remain unaccepted.
Authorization: The operator approved the unified design/plan, earlier bounded experiments and performance work, then the three remaining repairs and reasonable follow-up review fixes. This revision performs only those related repairs and bounded tests; no commit, push, dependency installation, version change, tag, release or new research training is authorized.

## 1. Intent, scope, and governing documents

Add optional cross-view Coordinate and NeCo-style neighbor-ranking objectives to pretraining. Corresponding locations in different augmented views of the **same image** should have similar backbone patch features. Local representation quality is valuable independently of global barcode improvements, but an engineering implementation is not evidence of morphological benefit.

This single document owns both objectives and their shared geometry-to-training handoff, configuration, initialization, recovery, optimization, reporting, and delivery boundaries. The algorithm chapters remain independent: ranking requires neither a positive Coordinate weight nor demonstrated Coordinate benefit. Independent enablement and staged acceptance do not require separate design documents.

The [master design](2026-03-29-otuformer-design.md), [training modularity design](2026-11-07-training-modularity-cli-testing-design.md), [spatial geometry/features design](2026-11-12-spatial-geometry-features-design.md), [relational FIFO design](2026-11-10-relational-fifo-design.md), and mandatory [testing policy](2026-11-07-development-testing-policy.md) govern ownership, compatibility, geometry, and verification. This work addresses roadmap 4.7, not master-design section 4.7. The ignored local roadmap is background, not an executable specification or public flag reference.

### Goals and anti-goals

- Deliver two independently selectable additive objectives through the existing pretrain loop, in staged implementations and acceptance.
- Preserve existing global/local objectives and exactly one existing `patch_loss` choice: `none`, `consistency`, `masked-feature`, or `ibot`. Coordinate and ranking are **not** additional mutually exclusive choices in that enum.
- Preserve augmentation profiles, defaults, orientation behavior, actual operation order, and ordinary tensor/RNG behavior when both objectives are off. Track all existing spatial operations rather than narrowing training to crop/resize.
- Reuse the actual feature grid and geometry primitives; do not add a second trainer, universal loss/plugin framework, teacher-local forward, foreground estimator, or duplicate split tool.
- No new Coordinate finetuning, perspective/elastic augmentation, segmentation requirement, disk patch bank, anatomical homology assertion, or automatic biological interpretation.
- No claim that a cosine loss alone prevents patch collapse, or that Coordinate is superior/inferior to DenseDINO without the relevant experiments.

## 2. Discussion decisions and alternatives

The operator selected these decisions during the 2026-11-15 discussion:

| Decision | Selected behavior |
| --- | --- |
| Scope | Coordinate and neighbor/ranking in the same effort and one design, independent enablement and staged delivery. |
| Coordinate | Direct normalized-feature cosine, an independent baseline rather than a DenseDINO reproduction. |
| Composition | Each new weight is independent, fixed within a run, finite and nonnegative; default zero. Either or both may accompany any existing patch mode. |
| Features | Both consume ordinary unmasked branches. Coordinate uses backbone features; ranking owns a separate student/EMA-teacher MLP. |
| Pairs | Ordinary teacher global A to student global B and every local; teacher global B to student global A and every local. Exclude identical views and other images. |
| Geometry | Actual student patch centers, valid support in both views, map to teacher and bilinear sample. |
| Sampling | Query default 49; reference default **16**, explicitly approved by the operator on 2026-11-15 after audit (superseding the initial reference default 49). No replacement or repeated padding; use actual counts below the caps. Default approval does not certify memory fit or approve the complete design. |
| Reference scope | The final consolidated proposal, explicitly accepted by the operator, uses the current ordered pair's aligned teacher-query pool across the batch, not all teacher patches and not a cross-pair shared pool. Per-sample independent reference draws remain an author detail. |
| Reduction | Mean points within sample, mean contributing samples within each ordered view pair, mean contributing pairs. No fixed global/local half-weight split. |
| Initialization | Separate new-run checkpoint initialization, mutually exclusive with resume, backbone only, default source EMA teacher with explicit student alternative; no silent fallback. |
| Optimization | Existing optimizer/LR/weight-decay policy, joint clipping, and teacher EMA. No separate NeCo learning-rate recipe. |

Approaches considered:

1. **Direct Coordinate plus independent NeCo ranking — selected.** Smallest route that tests explicit cross-view feature agreement separately from relational ordering. Their joint benefit remains a hypothesis.
2. **DenseDINO reference-token distillation.** A distinct feature construction and supervision mechanism, not interchangeable with sampling ordinary backbone patches. Not selected for this delivery; its intended inference remains unchanged, so rejection is not based on requiring a new exported embedding API.
3. **Make all dense objectives mutually exclusive patch modes, or add a general composition engine.** Rejected: the former blocks agreed combinations; the latter adds unsupported abstraction.

Builder-level choices below (field/flag names, state encoding, RNG stream policy, numerical guards, and failure details) are author specifications for review, not additional decisions attributed to the operator.

### Primary sources checked on 2026-11-15

Source URLs are reproducibility references, not dependency pins:

- NeCo paper: `https://arxiv.org/abs/2408.11054`.
- Official training/loss code: `https://github.com/vpariza/NeCo/blob/main/src/neco.py`.
- Official feature/head code: `https://github.com/vpariza/NeCo/blob/main/src/models/vit.py`.
- Named DINO configuration: `https://github.com/vpariza/NeCo/blob/main/experiments/configs/models/small/neco_dino.yml`.
- Sorter implementation: `https://github.com/Felix-Petersen/diffsort/blob/main/diffsort/diffsort.py`, `https://github.com/Felix-Petersen/diffsort/blob/main/diffsort/functional.py`, and `https://github.com/Felix-Petersen/diffsort/blob/main/diffsort/networks.py`.

The official code projects/normalizes patches before ROI alignment, forms a pair-aligned batch reference pool, compares soft permutation matrices, and offers a history-queue branch. Its named DINO config uses a three-linear 2048/256 head, bitonic cosine sorting with steepness 100 for both networks, 7x7 alignment, a queue and distinct head/backbone learning rates. The source class defaults are not the same named recipe.

This project preserves the ordering mechanism but changes geometry to checked student-center sampling, uses actual ragged counts without repeated filling, enables only current-batch references, combines existing SSL objectives and keeps their optimizer/EMA recipe. Call it **NeCo-style**, not an exact reproduction or a validation of joint from-scratch training. Original dense post-pretraining evidence is not a guarantee for this project's training mixtures.

## 3. Existing implementation and intended extension

At baseline `fd745ef`, `PretrainForward` / `OTUFormerEncoder.forward_pretrain` expose complete final-normalized tokens and a real two-dimensional grid. Global student forwards are already unmasked. For each global `s_out`, remove `backbone.num_prefix_tokens` exactly once from `s_out.tokens` and use `s_out.grid_size`. For locals, retain the second output of the existing `student(lv)` call: `OTUFormerEncoder.forward` has **already removed** CLS/register prefixes, so do not slice it again. Resolve its real grid through the existing checked patch-grid logic for that input and assert patch count equals grid height times width. Do not add a second local backbone pass or infer a grid with `sqrt(N)`. Tests must lock both entry-point contracts, including nonzero registers.

The existing patch modes remain unchanged:

| Mode | Input and objective | Relationship to the new objectives |
| --- | --- | --- |
| `none` | No existing patch loss. | Coordinate and/or ranking can still be on. Global/local SSL remains. |
| `consistency` | Visible same-crop/same-position cosine at selected positions; no input masking. | Not cross-view geometry consistency; may coexist. |
| `masked-feature` | Extra masked global student forward and predictor, with detached normalized final-four teacher target. | Keep its masked prediction branch; new objectives use ordinary unmasked tokens. |
| `ibot` | Extra masked global student forward with student/teacher prototype heads and patch center. | Do not use these logits as Coordinate or NeCo features. |

Ordinary teacher global tokens must be obtained for **both** globals whenever either dense objective is active. The current relational optimization can request only one ordinary teacher view; extend that requirement locally. Do not change relational weak-teacher targets, center population, CLS pair policy, or FIFO observations to obtain dense features. No teacher local is added.

## 4. Shared batch and geometry handoff

### Enabled-only dataset path

`MultiCropDataset` currently returns a list of view tensors or the FIFO-only mapping `{views, image_key}`. `_unpack_pretrain_batch` validates that mapping strictly. Both owners must change together, not silently accept arbitrary extra keys.

Author-specified envelope shapes:

| Dense enabled | Relational FIFO enabled | Batch fields |
| --- | --- | --- |
| No | No | Historical list of views, unchanged. |
| No | Yes | Historical `{views, image_key}`, unchanged. |
| Yes | No | `{views, geometry}`. |
| Yes | Yes | `{views, geometry, image_key}`. |

For ordinary global/local views, use each transform's existing `transform_with_geometry` so the image and trace share a single realization; never resample an augmentation to recover its geometry. For the enabled path, use a narrow collator: collate tensors/keys by the existing behavior and retain per-view/per-sample serialized traces as ordered ragged records rather than default-collating variable operation sequences. The geometry list follows view order: ordinary global/local entries hold batch-row-aligned traces; unused weak-teacher view entries are explicit `None` slots. Weak views retain their existing tensor-only transform and RNG path: do not compute, serialize or collate traces/support for them. Reject unexpected `None` on an ordinary view, a trace on a weak-only slot, missing/extra fields, malformed sizes and row/view misalignment. This keeps a single view-index contract without manufacturing unused geometry.

Use the already resolved local-view count and weak-view enablement to validate that layout, rather than trusting metadata to label its own None slots. Check tensor/trace canvas sizes and same-row source sizes; collation preserves original sample/view order. This is structural validation, not detection of an externally forged but internally consistent trace without an identity binding. Do not add image-content hashing to the dense-only path. The ordinary startup/initial-forward probe may validate metadata, but must not draw dense queries/references or count a training contribution.

Do not request traces, support masks, new collation, or dense sampling when both weights are zero. FIFO-off dense training does not thereby acquire FIFO data hashing or keyed-identity requirements. FIFO-on runs retain all their existing data locks and restore checks.

### Geometry contract

Reuse `vision.geometry.GeometryTrace`, realized operations and their ordered clipping/support rules, `patch_support`, and `vision.features.sample_grid_features`. Preserve the existing support-version identity in the enabled recipe. Build pixel validity from `pixel_support(trace)` and feed it to `patch_support` with the verified grid/patch dimensions, reusing each view's mask across pairs. Source-to-view matrices map continuous image-edge `(x,y)` coordinates, array indices are row-major `(y,x)`, and pixel centers are `(column+0.5,row+0.5)`. Store explicit dimensions and patch/grid sizes.

For student point `p_s`, compute `p_t = M_t inverse(M_s) p_s`. A final invertible affine map alone does not restore content lost at intermediate crops or canvas truncations. Check geometric survival and conservative image-resampling support through the entire recorded sequence on both views. Padding/fill is not background evidence; support is not a foreground mask. Full patch validity does not remove ViT attention influence from nearby invalid tokens.

Use the existing bilinear sampler, `align_corners=False`, and its complete-support test: every nonzero interpolation contributor must be in bounds and valid. Zero-weight out-of-grid neighbors do not invalidate an exact edge center. Keep coordinate construction/inversion metadata in CPU float64, but qualify eligibility under the sampler's float32 representation/nonzero-contributor convention; transfer coordinates after CPU conversion to float32, with the Boolean validity mask on the feature device. Do not place float64 metadata on devices that do not support it or let different precision rules silently choose different contributors. For patch width/height `(pw,ph)`, feature indices are `(x/pw-0.5,y/ph-0.5)`. The 64x64 input, 16x16 patch fixture maps `(24,40)` to feature index `(1,2)` and normalized grid `(-0.25,0.25)`.

### Eligible points and shared sampling

For each ordered view pair and image, enumerate actual student grid centers in row-major order, then retain only centers with complete student and mapped-teacher support. No ROI bounding rectangle may substitute for a rotated/clipped overlap polygon. No overlap-percentage threshold or guaranteed minimum crop intersection is added; one valid point is sufficient for Coordinate.

Uniformly sample without replacement up to `dense_query_points` (default 49); retain row-major output order after selection. Below the cap, use all points and consume no sampling draw. These are caps, not fixed counts: a 64px view with 16px patches has at most 16 centers; the current 96px local default has at most 36, before validity/overlap filtering. Do not reject additional images or duplicate points to create a fixed shape. Both enabled objectives share the selected coordinates and geometric mask; their feature spaces and loss-availability rules remain separate. Ranking-only obtains the same handoff without executing Coordinate.

Author RNG choice: use independent CPU `torch.Generator` streams for query selection and ranking references, seeded from the existing run seed and distinct versioned stream identifiers. Save their states and restore before the next selection. Ranking reference draws must not advance the query generator; changing which objective is off does not alter the augmentation RNG stream through sampling. Draw consumption is **data-dependent**: candidate counts at/below a cap take no draw, while exceeding it takes a draw. Exact resumed selection therefore requires the same eligible sets, reference-pool order, batch/view order and caps as well as the restored generator state; matching a seed or loader length alone is insufficient. Dense-only recovery does not promise changed-data replay or silently acquire the FIFO's dataset fingerprint. Head initialization may legitimately alter new-run RNG when ranking is on; make no cross-recipe trajectory-parity claim. Do not instantiate these generators on the all-off path. Worker-produced traces travel with their images; sampling runs in the training process.

## 5. Coordinate feature path and exact objective

Use the final complete-block backbone tokens **after the existing backbone final norm**, without CLS projection, masked-feature predictor, or iBOT head. Reshape to the verified grid. Student features are read at selected exact student patch centers; teacher raw features are bilinearly sampled at mapped points, then both vectors are channel-L2 normalized. This is an explicit author baseline choice: interpolate the existing backbone vector field first, preserving contributor magnitudes in the linear mixture, then compare its direction. It is not attributed to DenseDINO or claimed optimal. Pre-normalizing each neighbor instead interpolates unit directions and changes the target, so do not silently substitute it. Ranking intentionally preserves its separate project/L2/sample/renormalize recipe below.

Let `a_bvq` and `t_bvq` be those unit features for image `b`, ordered view pair `v`, and selected point `q`. Teacher computations are detached/no-grad.

```text
ell_coord(b,v,q) = 1 - dot(a_bvq, stop_gradient(t_bvq))
l_coord(b,v) = mean_q ell_coord(b,v,q)
L_coord(v) = mean_b l_coord(b,v) over samples with at least one valid point
L_coord = mean_v L_coord(v) over ordered pairs with a contributing sample
L_total = L_existing + coordinate_weight * L_coord + ranking_weight * L_rank
```

`L_existing` retains current global/local, selected patch loss, and any enabled KoLeo term; no coefficient, normalization, teacher center, mask ratio, or patch mode is reinterpreted. With L locals and both globals, there are 2 global-to-other-global and 2L global-to-local ordered pairs. At the default L=6, all-valid pair weighting is therefore 12:2 = **6:1**, or 6/7 local and 1/7 global, for each new objective. These are **pair-level objective shares, not point-count shares**. With patch16, query cap49, global224 and local96, per-image all-valid query counts are `2*49 : 12*36 = 98 : 432`, approximately **18.5% global / 81.5% local**; these counts do not override the selected hierarchical reduction. Both shares change when pairs or samples lack valid overlap. This local-heavy bias is the explicitly selected pair policy, not accidental balancing. Missing overlaps alter contributing denominators and can leave a pair supported by few points; log actual point/sample/pair counts by branch. Each objective computes its own denominator, so missing ranking references cannot remove a sample from Coordinate.

Author numerical rules: reductions and normalization checks use float32. Require every participating feature to be finite with channel norm > `1e-6` before L2: selected student centres and each teacher contributor whose bilinear weight is positive. Unselected patches, empty rows and exact-zero-weight neighbours are outside this numerical population. Also check interpolation cancellation: for nonnegative nonzero bilinear weights w_i (sum 1) and contributing vectors f_i, define `m = sum_i w_i*f_i`, `s = sum_i w_i*norm(f_i)`, and `rho = norm(m)/s`. Require `s > 1e-6` and `rho > 1e-6`; exact-center sampling has rho=1. Apply this to Coordinate's raw teacher interpolation and ranking's pre-normalized projected teacher interpolation. Check original projected vectors before their initial L2, too. These are **provisional author numerical thresholds**, not calibrated representation-quality thresholds or results of an extreme-value analysis. Small CPU fixtures establish expected behavior, not long-run failure probability. The planned qualification sequence below must report minimum participating norm/rho and numerical failures before the versioned recipe freezes these thresholds. A failed finite/norm/rho check aborts the step with pair/sample context; it must not silently remove difficult points from the geometric denominator. Log minimum participating norm and rho from detached cached values. A rho of `1e-3` passes this numerical guard and is **not** thereby a validated directional target. If first-run diagnostics show a substantial recurring population below an indicative rho of `0.1`, review interpolation geometry, precision and feature statistics before accepting the recipe; one low minimum alone does not establish that population. The value 0.1 is a research-review trigger, not a hard guard, calibrated quality boundary or permission to change thresholds automatically within a run. Any revised guard is a versioned recipe decision followed by requalification; do not silently relax it or filter failing points. The absolute epsilon alone is not an anti-cancellation or anti-collapse test, and a cosine derivative contains an inverse feature norm, so it is not uniformly bounded as that norm approaches zero. No extra cosine clamp or factor of two is introduced. If all geometry pairs are unavailable, the zero contribution is reported as skipped, not observed perfect consistency. Nonfinite new losses or gradients abort before optimizer/EMA updates.

Constant identical unit features give zero Coordinate loss. A less extreme shortcut is spatial over-smoothing: making neighboring feature directions similar can reduce interpolation-based mismatch without preserving discriminative local structure. The numerical cancellation check does not detect that shortcut. Record spatial neighbor similarity/feature diversity together with matching precision in separately authorized research evaluation, not just loss reduction. Nearest-neighbor targets, pre-L2 interpolation, and restricting subpatch offsets are potential **new research variants**, not silent runtime fallbacks or features in this delivery; each would require its own explicit recipe/approval. EMA/stop-gradient and active global SSL do not mathematically guarantee patch diversity or establish that smoothing is absent.

## 6. Independent training feature space

Let backbone width be D. The fixed version-1 student head is:

```text
Linear(D,2048,bias=True) -> GELU ->
Linear(2048,2048,bias=True) -> GELU ->
Linear(2048,256,bias=True)
```

Author initialization follows the checked head's linear initialization: truncated normal weights, standard deviation 0.02, zero biases. No batch normalization, dropout, prototype layer, centering, Sinkhorn, prediction head, or shared CLS projector. The teacher head has the same architecture, starts as an exact copy on a new run, has no optimizer parameters/gradients, and uses the existing teacher EMA momentum after successful steps. On resume, restore the two saved heads separately without synchronizing them.

The input is final-normalized complete-block patch tokens with all prefix/register tokens excluded. Compute this field in order:

```text
backbone final norm -> per-patch MLP -> per-patch channel L2
-> checked geometry sampling -> cosine channel renormalization
-> reference similarities -> differentiable permutations -> loss
```

Student queries are exact grid-center samples; teacher queries are generally bilinear samples. Apply the nonlinear MLP **before** interpolation; projecting an interpolated backbone vector is not equivalent. Preserve pre-sampling L2 as part of this recipe, unlike Coordinate's sample-then-normalize raw backbone path. Use float32 feature normalization, cosine, sorting and reductions. Reuse the complete-support mask and the common absolute-norm plus relative-cancellation guards from the Coordinate numerical contract; do not use zero-padding as evidence or confuse numerical failure with missing geometry.

Each view's projected field may be reused across multiple pairs without changing autograd. Implementation may restrict projection to needed patch contributors if mathematically and numerically equivalent; it must not reorder projection and interpolation. Reusing backbone forwards does not make MLP activations or local-backbone gradient retention free.

For standard embedding/spatial extraction and evaluation, read the ordinary backbone/current CLS contract, not this head. Training-only head keys live outside the encoder state dictionaries, avoiding accidental exported-embedding identity changes.

## 7. Pair-conditioned current-batch reference pool

Use the common ordered view-pair enumeration and sampled coordinates. An ordered pair v is a teacher-global index and a distinct student view index, applied across batch rows. For each row b, the handoff supplies Q_b valid selected student centers (0 <= Q_b <= query cap) and mapped teacher points.

For a fixed v:

1. Obtain projected, sampled and cosine-renormalized teacher vectors at all selected corresponding points for all rows with Q_b > 0.
2. Detach these vectors and concatenate in batch-row then query row-major order into U_v. A row has at most 49 entries by the initial query budget. The pool is bounded by batch size times query cap; do not silently expand it to every raw teacher patch or mix teacher A/B and unrelated pair overlaps.
3. Let N_v be the pool size and R_v = min(reference cap,N_v). For **each contributing sample b**, uniformly sample R_v distinct pool entries without replacement, keeping the sampled order. Below/equal to the cap, use the full pool in canonical order with no extra draw. Separate samples may choose overlapping lists.
4. All student queries in sample b and their corresponding teacher queries use exactly this reference list and ordering. Do not independently shuffle teacher references, use student reference vectors or backpropagate through the pool.
5. Same-image and self references are allowed. Do not compare filenames or infer specimen identity. Distinct coordinates with equal vector values remain distinct entries; uniqueness is by pool entry, not feature-value deduplication. Never synthesize/repeat entries to reach R_v.

The pool is **the teacher side of this ordered pair's selected correspondence/query set**, not every teacher grid patch. This matches the pair-conditioned scope of the official batch-only branch (`aligned_emb_gc_sel`), which is selected for the current student view before flattening across batch; it is not a reference pool shared across all view pairs. Our differences are actual patch-center queries instead of a fixed ROI grid, ragged valid counts, bounded independent per-sample draws rather than the upstream shuffle/repeat construction, and no history queue. Those sampling changes affect the empirical reference distribution and must appear in experiment provenance. The pair-conditioned scope was explicitly accepted in the final discussion; the per-sample independent drawing rule remains an author detail. Use the independent reference RNG. No cross-step references, transient reference serialization, label use, object/part filtering, or all-gather is added.

If Q_b=0, that sample is unavailable. If N_v<2, ranking for v is unavailable even if Coordinate can use a single point. A batch of one image is allowed; it can form a valid pool of two or more points from that image. Fewer than two distinct pool entries never triggers replacement, cross-pair fallback or a queue fallback. Publish the actual population and reference count.

## 8. Exact differentiable ordering and loss

For sample b, query q and reference entry j, let z_s, z_t and r_j be normalized projected student, teacher and detached reference vectors. Define:

```text
c_s[j] = dot(z_s, r_j)
c_t[j] = dot(stop_gradient(z_t), r_j)
(_, P_s) = DiffSortNet(bitonic, size=R, steepness=100, distribution=cauchy)(c_s)
(_, P_t) = same fixed sorter recipe applied to c_t under no_grad
```

Use **ascending cosine** on both sides; nearest entries occupy final ranks. Do not negate only one side, use Euclidean distance, hard `argsort`, similarity softmax, or another distribution under the same recipe ID. The convention is `P[j,k] = input reference j assigned to sorted rank k`, with row-vector `sorted = c @ P`. A concrete four-element fixture is `c=[0.4,-0.7,0.9,-0.2]`: the hard-limit rank order is `[1,3,0,2]` and the input-by-rank matrix has rows `[0,0,1,0]`, `[1,0,0,0]`, `[0,0,0,1]`, `[0,1,0,0]`. On the checked float32 Cauchy/100 source the soft sorted vector is approximately `[-0.6904513,-0.1968169,0.3968168,0.8904513]`; `c @ P` matches within `6e-8`, while `P @ c` does not. Freeze an equivalent fixture for the qualified package; tests verify an already defined convention, not decide it later.

Use **unclamped** `log(P_s)`, matching the checked upstream loss. Validate finite student probabilities strictly >0 and teacher probabilities >=0, and valid row/column sums (provisional float32 absolute tolerance `1e-5`) before reduction. This tolerance has comfortable margin in the small source-snippet fixtures (observed errors were below `5e-7`), but is an **author-selected conservative candidate**, not a bound derived from extreme distributions or evidence of long-run/device safety. Abort rather than silently clamp, renormalize or invent a target if validation fails. The previous draft's `clamp_min(1e-12)` is removed: increasing the floor to `1e-7` would widen its zero-gradient region, not fix it. Positive Cauchy comparator coefficients in exact arithmetic do not guarantee that every product stays nonzero under every float32 network/size; qualify extremes and report observed minima. Tied inputs remain soft ties; no random jitter.

The per-query loss, with P shaped `[input_reference,sorted_rank]`, is:

```text
ell_rank(b,v,q) = -(1/R_v) * sum_j sum_k
                    stop_gradient(P_t[j,k]) * log(P_s[j,k])
               = mean_j[-sum_k P_t[j,k] * log(P_s[j,k])]
               = R_v * mean_all_matrix_cells[-P_t * log(P_s)]
```

Thus **retain `1/R_v`**: it is the single mean over reference rows, not a second normalization. Removing it multiplies the objective by R_v; an unscaled mean over all R_v squared cells divides it by R_v. For each query, transposing both matrices and using the upstream summed axis followed by the remaining-axis mean gives this same scalar; unit row/column sums do not eliminate the factor R_v. Teacher and reference tensors are detached. Mean queries per sample, then contributing samples per ordered pair, then contributing pairs. Do not silently point-weight large overlaps or rebalance global/local groups.

Teacher soft-target entropy can make an identical student/teacher ranking loss positive. A nonzero value is not necessarily disagreement; zero is not its required optimum. Do not replace cross-entropy with KL to force a zero baseline. Both Coordinate and ranking may be present:

```text
L_total = L_existing + coordinate_weight * L_coord + ranking_weight * L_rank
```

Coordinate lies in [0,2] in exact arithmetic. Ranking cross-entropy is **not bounded by log(R)**: log(R) is the uniform-prediction value and the maximum categorical entropy, not an upper bound for mismatched predictions. Near-zero student probabilities can make CE much larger. Never infer gradient dominance or a universal 5–20x ratio from these scalar losses. The weights are not automatically calibrated; ranking alone does not mathematically guarantee no collapse, and CE against ambiguous constant orderings is not evidence of useful neighborhoods. The normal logger records cached raw/weighted terms, not per-objective gradient attribution; the separately authorized diagnostic boundary is specified below.

## 9. Optional dependency and bounded sorter lifecycle

Use the external `diffsort` package, not an in-repo reimplementation or a new training framework. Proposed extra name is `ranking`. Do not make `diffsort`, Lightning or NeCo's experiment environment mandatory core dependencies. Import and construct it only when the resolved ranking weight is positive; keep schema/help/checkpoint read-only consumers independent of it.

Before output mutation, enabled preflight checks the optional import/API and the selected device's actual small forward/backward, including float32 cosine vectors and a non-power-of-two reference count. If unavailable/unsupported, fail with the install-extra/device guidance; do not silently switch objective, device, sorter or reference count. A tiny dependency probe must preserve caller RNG and allocate no model or dataset. Dependency installation/download is not authorized by this draft.

No compatible package version is certified by reading source. The execution plan must identify a candidate exact release/range, inspect its packaged code/license/API, and schedule isolated package/dependency-floor/selected-device qualification before changing the extra requirement. Record tested library version and the qualified loaded-source identity in runtime provenance and checkpoints. The plan separates enforceable runtime version/source hashes from the audited wheel hash, which is artifact provenance only; do not require recovery or download of an installed wheel archive on resume. A loaded implementation whose source bytes cannot be verified is unsupported for enabled ranking under this policy (a source-readable zip loader is not automatically unsupported); disclose this distribution limit. Coordinate/off and ordinary read-only consumers remain independent of it. A version mismatch on enabled resume must fail unless that release is explicitly qualified as numerically compatible with the saved recipe; do not assume a version string alone proves equivalence. No floor/range is invented here.

`DiffSortNet` fixes reference-list size per instance and holds device-specific network tensors. Bucket/group by actual R_v and reuse at most the necessary fixed-size sorters for the active device/dtype/recipe; retain no autograd/query/reference graphs in the cache. Non-power-of-two support must be proven on the chosen release. Do not pad to 64 or repeat references to avoid building a sorter.

Author memory policy: cache only sort-network constants in a maximum-four-size LRU; retain no query/reference graphs. Reuse matching student/teacher constants. **Ordinary forward chunking followed by one total backward does not bound retained autograd memory**: all chunk graphs remain live. It may only reduce temporary workspace; do not advertise it as activation-memory relief. Split backward would change the approved step mechanics and is not selected here. Qualified activation recomputation would need separate numerical/cost validation; do not silently introduce it.

A source-snippet CPU check on 2026-11-15 (torch 2.12.1, four float32 query rows, Cauchy/100, selected **unclamped row-CE**, unique saved-storage accounting excluding inputs, teacher targets and network constants) retained 240,908 bytes/query at R=49 and 13,760 at R=16. One versus four forward chunks retained the same totals: 963,632 versus 55,040 bytes respectively for four rows. The operator-supplied third-round audit reports an independent implementation yielding 231,108/12,736 bytes/query, about 4.1%/7.4% lower. These are **reviewer-reported results**, not another run verified here. Different saved-storage inclusion boundaries can explain such differences but require matching instrumentation to establish; the under-8% reported spread does not alter the order-of-magnitude budget concern. It is not a confidence interval, an exact reproducibility guarantee, or GPU evidence.

For the repository's actual new-run size/count defaults (B=32, patch16, global224, local96, six locals), the **all-valid counting upper case** is `32*(2*49 + 12*36) = 16,960` queries/step. Linear extrapolation of that measured CPU saved-storage rate gives about **3.805 GiB at R=49 versus 0.217 GiB at R=16** for the checked sorter/log-loss graph alone. Real geometry usually lowers eligible counts; changing backbone/patch size or view settings changes this arithmetic. These are extrapolations from a source-snippet CPU check, **not measured default GPU peaks**, and exclude backbone/head/reference activations, backward workspace, constants, allocator and existing objectives. They motivate a more conservative initial R, but do not prove that 16 is necessary or sufficient on any hardware. At first enabled qualification, report **actual queries/step and actual reference counts alongside the matching configured counting bound** (16,960 for the defaults), separated by ordered pair/global/local and measured over the declared steps. Recompute the sorter-only extrapolation from actual counts and report measured total peak separately; never present this all-valid bound as the expected population under random rotation/cropping. The larger stress example B=64 with 49 queries for every one of 14 pairs would instead extrapolate to about 9.85/0.56 GiB; it is not the repository default.

Operator-approved default revision (2026-11-15): retain query default 49 and lower the reference default from the initially discussed 49 to **16**. Preserve 49 as the maximum selectable reference cap in this first recipe, not as a certified safe setting. **Statistical cost:** fewer distinct references change the competitors and soft-permutation target distribution, finite-sample variability and entropy/loss baseline. This is not merely a memory-equivalent implementation. There is no established universal `1/sqrt(R)` variance law or 1.7x noise factor for this nonlinear, dependent ranking statistic. Across different reference caps, do not treat equal raw/weighted loss values or the same numeric ranking weight as calibrated equivalents; retune/compare under controlled protocols and record actual reference-count distributions. The cap and sampling policy belong to the checkpoint recipe identity (already required below); changing them requires a new run. Actual R can still vary within an unchanged recipe when the valid pool is smaller, and its population/denominators must remain visible. Bound query cap to 1..49 and reference cap to 2..49 to reject e.g. 4096 before allocation. These are scope/memory-containment ceilings, not hardware-fit guarantees. The operator explicitly approved this default change; complete-design approval and empirical memory qualification remain separate gates. Before implementation acceptance, measure a real installed-package tiny loop plus the intended batch/global/local/patch/SSL combinations, peak training memory and first/steady-step timing on the selected device. No production-default fit claim is allowed until that gate passes. If it fails even at 16, stop and obtain a revised budget or separately approved memory strategy; do not silently shrink counts, switch device, or change backward semantics.

## 10. Configuration and applicability — author surface

Proposed pretrain options (names are author specifications, not a current CLI reference):

| Option | New-run resolution | Applicability |
| --- | --- | --- |
| `--coordinate-weight` | `0.0`, finite and >=0 | Positive enables Coordinate. |
| `--ranking-weight` | `0.0`, finite and >=0 | Positive enables the ranking objective. |
| `--dense-query-points` | `49`, integer 1..49 | At least one new objective active; cap, not guaranteed point count. |
| `--ranking-reference-points` | Operator-approved default: `16`, integer 2..49 | Ranking active only; recipe-identity field, not a calibrated cross-R loss scale. Complete-design approval and memory qualification remain outstanding. |
| `--init-checkpoint` | Omitted | New run only, supported OTU SSL checkpoint. |
| `--init-source` (`teacher` or `student`) | `teacher` | Only with `--init-checkpoint`; selected branch must exist. |

There are no additional enable flags: a positive weight enables its term. Omitted inactive settings have no effect; reject explicitly supplied inactive sampling/reference/source options, including inactive explicit defaults. Resolve requested, effective, inherited, and inactive values once before output mutation; do not rewrite raw request provenance to look inherited by default. Run applicability is both `ssl_method=relational` and `dino`, all four patch modes, optional KoLeo, and the existing relational FIFO where that method permits it.

Weights stay fixed over the run; no auto-balancing or objective warmup is added. New checkpoint-initialized runs resolve objective settings as new requests/defaults, not as inherited source training recipes. A positive ranking weight triggers dependency checks from the optional-dependency chapter; Coordinate-only and all-off invocations must not import `diffsort`.

## 11. Backbone-only initialization of a new run

This independent entry point is not resume and may be used whether or not a new dense objective is active. Its implementation can be scheduled independently of Coordinate/ranking; record the one chosen source/test sequence at execution time rather than conflating evidence from alternative task orders. It creates a fresh run/output identity, starts at epoch/step zero, and records the immediate source. It never mutates the source run.

**Scope justification:** roadmap 4.7 does not intrinsically require a new initialization API. The operator nevertheless explicitly selected it (Q11/Q12 and the final consolidation) to start a new objective stage from an existing OTU backbone, because strict `--resume` deliberately cannot add/change objectives. Without it, the delivered post-pretraining use case would require an unsupported manual checkpoint edit or another timm-started run. It therefore remains an explicit, independently testable subtask in this one approved scope, not an inferred prerequisite or general checkpoint-conversion product. Supporting all-off initialized runs follows the same source-transfer path and does not authorize extra consumers or metric/third-party imports.

- `--init-checkpoint` and `--resume` are mutually exclusive before output-directory creation/overwrite. Explicit `--init-source` without a source is invalid.
- Accept recognized OTU SSL encoder payloads for relational or DINO, including recognized historical SSL layouts, but **require the exact selected `teacher` or `student` branch**. Do not use the shared reader's permissive branch fallback. Metric/finetune, ONNX, raw third-party backbone-only and unsupported SSL-version payloads are outside this first contract.
- Use existing checkpoint architecture/size/register resolution. Omitted architecture/global size/register settings resolve from the source; explicit incompatible model, patch lattice, width, register layout or native position-grid requests fail. Reuse existing dynamic positional interpolation for legitimate local sizes; do not resize saved position tensors or fabricate registers during transfer.
- Validate the source checkpoint's recognizable SSL identity, complete selected projector/encoder shape/finiteness and complete backbone state before extracting the backbone. For recognized historical layouts without a recorded model name, the plan passes the caller's existing model fallback explicitly; recorded metadata remains authoritative and no alternative state branch may be substituted. New-run loading copies only `backbone.*`, including position and register tensors. Validate exact target backbone keys/shapes and load strictly. A malformed source projector/identity is not silently laundered merely because only backbone weights transfer.
- Rebuild all current-run training heads, masks, centers, optimizer, schedules, and relational history. Preserve the usual selected-method initialization for new CLS heads; synchronize the complete newly initialized student encoder to teacher once before the first forward. Ranking creates and synchronizes its own head pair. Do not copy source optimizer/progress, head states, centers, queues, or RNG.
- Do not perform a timm pretrained download when reconstructing a fully supplied source backbone. The existing new-run path without `--init-checkpoint` remains unchanged.
- Record source resolved path, SHA-256 of the actual file loaded, selected state key, source active SSL identity, architecture and the `backbone-only` transfer policy. Reuse existing hashing/identity helpers; read/hash one stable source snapshot or detect concurrent changes before accepting it.

Author provenance choice: a versioned `config.pretrain_initialization` bundle owns the **immediate** initialization source, not inherited finetune `ssl_source_*` identity or a claim that the old projector was transferred. Keep the current run's active `ssl_method`/`ssl_head` authoritative. On resume, retain that recorded source without reopening or requiring the original file. Ordinary downstream readers consume current encoder weights and must not confuse initialization provenance with current architecture.

## 12. Enabled checkpoint encoding and strict recovery

Author format: when either new objective is active, persist `config.dense_training_config` version 1 and top-level `dense_training` version 1. They are training-only additive bundles; do not add a new backbone/head identity or alter the existing relational-target versioned key set.

The effective config contains both weights, query cap, geometry/support/sampling/reduction/feature policy identifiers, enabled objective identifiers, and the ranking fixed recipe/reference cap when ranking is active. It also records enabled-run batch size and steps per epoch for recovery validation. Names and exact serialized key sets must be frozen in the implementation plan before code is written; the plan may not change the semantics here. Reject unknown versions, missing fields, unexpected fields in these bundles, request/config/state contradictions and unsupported recipes rather than guessing.

The state contains query RNG, optional ranking-reference RNG, completed-step consistency information, and separate ranking head state defined below. Save at the existing supported **epoch checkpoint boundary**, coordinated with encoder/patch/DINO states, optimizer, schedules, centers and existing RNG. This is not mid-epoch replay support. Sort-network caches and transient aligned reference tensors are not training state.

Enabled recovery must:

1. Inherit omitted weights/caps and reject any explicitly conflicting values, including attempts to enable a previously disabled term or disable an enabled one. A historical absent bundle means both disabled; it cannot be upgraded by resume.
2. Require a complete config/state pair and exact recipe identity. A Coordinate-only saved run must not acquire a ranking head or reference RNG on resume.
3. Validate student/teacher backbone and current CLS/patch/DINO objectives independently, complete keys/shapes/finiteness, usable optimizer groups and per-parameter AdamW state, centers, schedules, epoch/step and applicable RNG. Extend existing strict helpers rather than copying a second implementation. Inactive heads on all-skipped ranking steps may legitimately have no optimizer state; validate this case explicitly rather than inventing moments.
4. Preserve enabled-run effective batch size and loader length; do not silently resize the ranking population or change pair/sample denominators. Retain the FIFO's stronger fingerprint/identity checks when its queue is also on; dense training alone does not add that fingerprint.
5. Restore dense generators without consuming draws, and preserve existing Python/NumPy/torch streams. Use the existing target-aware strict recovery checks for CUDA where required; do not newly seed missing device state. CPU/worker-zero controlled replay is the exact test boundary. Multiprocessing and cross-device numerical replay are not guaranteed.
6. Keep the resumed student and teacher distinct; never synchronize them during recovery. Validate all conditions before output/log/CSV mutation.

When both weights are zero, omit both dense bundles and do not instantiate dense modules/generators; raw requested default options in `args` are not enabled-state evidence. An initialization-only run may still record `pretrain_initialization` without dense state.

Enabled training checkpoints require a dense-capable executable for resume. Older readers may ignore additive state; successful embedding extraction is not proof that an older trainer can resume it. `extract`, spatial extraction, export, CAM and finetune initialization must retain current encoder/head resolution and output meanings without loading training-only ranking heads.

### Ranking-specific saved state and skipped-step accounting

When ranking is enabled, store complete student-head and teacher-head state, the reference RNG and an authoritative count of successful optimization steps on which ranking contributed. The heads are required even if every ranking step was skipped. A zero contribution count means no head optimizer entries/moments have been created; reject fabricated complete or partial moments. Positive participation requires valid per-parameter steps and moments consistent with the existing optimizer and equal to that head's contribution count, not the overall step count. On later unavailable-ranking steps, prior head moments/counters stay unchanged. Increment only after a successful contributing step. Never initialize fake moments to make malformed recovery pass.

Transient references, projected grids, query autograd graphs and sorter caches are not persisted. On a new checkpoint-initialized run, old ranking heads are discarded together with other old training heads; on resume, the original two heads are restored independently. Both relational and DINO may carry this training-only state; relational FIFO retains its existing method restriction and separate serialization.

## 13. Optimization, failures, and observability

Use the existing one-backward/one-optimizer-step path and joint gradient clipping. Add new trainable parameters to the deterministic optimizer groups. For each of the three NeCo student Linear layers, **weights use effective weight decay; biases use it too under the current default `norm_bias_weight_decay=True`**. A new run's current declared `--weight-decay` default is **0.05**, so all three weights **and biases** receive 0.05 unless configured otherwise. Mechanism: the default-policy group is `{'params': params}` with **no group-level `weight_decay` key**, so it inherits the constructor's `AdamW(..., weight_decay=args.weight_decay)`, not an implicit AdamW library default or hardcoded group value. Preserve this handoff, and test a nondefault `--weight-decay` as well as 0.05 so new head groups cannot silently shadow the effective coefficient. Explicit group coefficients under the opt-out policy must use that same resolved coefficient, with biases set to zero. With the existing `--no-norm-bias-weight-decay`, weights still receive the effective coefficient (default 0.05) and all three biases receive exactly 0. The named upstream NeCo DINO config instead starts decay at **0.04**, schedules it toward 0.5, and exempts biases; this project retains its existing coefficient/schedule policy rather than importing that recipe. Effective resumed values remain authoritative, not a fresh forced 0.05. Teacher-head parameters are never in optimizer groups. This explicit shared-policy choice was approved; it intentionally differs from upstream NeCo's bias-exemption default and must not be silently replaced by a head-only exemption. Do not add an optimizer, head LR, schedule or extra backward. On enabled dense steps, explicitly clear with `set_to_none=True` so a previously participating head can become disconnected without a stale zero/old gradient causing decay or moment updates. If an optimizer produces nonfinite updated parameters, abort before EMA/center/FIFO/checkpoint commit even without FIFO; do not auto-retry or imply that the mutation was rolled back. Teacher targets/reference features are detached. EMA occurs only after a successful optimizer step using the existing schedule. Queue commit order and DINO prototype freeze rules stay unchanged.

Expected geometry unavailability contributes no point/sample/pair to that objective's denominator. Ranking can independently be unavailable for insufficient references. Log these states; do not fail a valid mixed-objective step solely for empty overlap. Conversely missing metadata, nonfinite tensors/loss/gradient, invalid feature norms, dependency failures and corrupt checkpoints are errors, not skips. The existing objectives continue to supply training signal when new terms are skipped. Do not optimizer-step a disconnected ranking head merely to create AdamW moments on an entirely skipped ranking step.

Emit a resolved dense config once, with applicability and requested/effective/inherited sources. Log at the existing interval from cached detached step results, without extra forwards or diagnostic backward passes:

- raw and weighted loss per enabled objective, and total existing/new loss contribution;
- selected/eligible query counts, contributing/skipped samples and ordered pairs, separated global/local;
- actual ranking reference counts, shortages and objective-specific skip reasons;
- attempted and contributing denominators, so a zero contribution is distinguishable from an observed zero loss;
- minimum participating pre-L2 feature norm and interpolation cancellation ratio from already computed values, without a diagnostic backward. `dense_min_norm` is the Coordinate minimum across selected raw student-centre norms, raw teacher positive-weight contributor norms, and the teacher interpolated mixture norms before L2, reduced across contributing rows/pairs. It is not solely a contributor minimum or solely an interpolation minimum. Empty Coordinate populations leave it blank. Ranking projected-space diagnostics remain separate.

Empty-population losses/norm/rho statistics are unavailable (`None` internally and empty/inapplicable in CSV), while genuine count0 remains numeric. Do not manufacture measured zeros to fit a numeric-only result type.

Per-objective **backbone gradient norms are not obtainable from cached detached losses**. Separating them requires extra reverse-mode passes/VJPs, often retaining the graph; one existing total backward cannot recover the separate contributions. Do not add that work to normal logging or describe it as zero-cost. A separately authorized short diagnostic experiment may measure raw and weighted norms on the same declared backbone parameter set, before joint clipping, along with pairwise gradient agreement and measured overhead. Use those measurements to propose starting weights in a recorded experiment; no universal 0.1–1 prior or inferred equivalence from scalar CE/cosine is adopted, and no automatic runtime calibration is added. Reporting the **observed raw-loss scale** for an enabled objective is not a calibrated prior and is expected: the command reference states the measured magnitudes, the ratio to an existing loss term and a provisional starting band with its basis, and each enabled run logs its own raw/weighted value so an operator can check the ratio rather than trust a table.

All-off old-schema logs must remain compatible. When extending instant/epoch CSVs, recognize the **actual current header** and its supported predecessors; migrate old rows with empty/inapplicable new fields and preserve values. Unknown/duplicate headers fail unchanged. Disabled metrics are unavailable, not fabricated observed zeros. Record available per-stage startup/first-batch/steady-state/evaluation timing and memory evidence before optimizing; do not claim local token reuse makes gradient retention or sorting free.

### Reporting surfaces: CSV versus the training-curves PDF

The instant CSV and the training-curves PDF are two different products of the same
logger, and neither implies the other:

- `logs/instant_metrics.pretrain.csv` carries the dense columns (raw and weighted
  losses, eligible/selected queries, contributing samples and pairs, reference
  population and reference count, shortage samples, minimum norm, minimum rho and
  minimum student probability). An unavailable statistic is an empty cell, never a
  measured zero.
- `logs/training_curves_pretrain.pdf` is produced by the plotting step from that CSV.
  An enabled run appends one diagnostics page to the existing page; an all-off run
  keeps exactly the historical single page. The dense page draws the two raw losses
  beside the existing total, the two weighted contributions, the query population,
  the contributing denominators, reference population/count with shortages and
  skipped log points, and separate panels for minimum norm, minimum rho and minimum
  student permutation probability, because those scales are not comparable.
- Writing a row to the CSV is **not** evidence that a panel exists in the PDF. The
  panels exist only when the dense columns are present *and* the plotting step runs
  for that run; a reader must verify the file, not infer it from the CSV.
- Plotting never fabricates a series: an all-missing column is skipped rather than
  drawn as a flat zero, and legend labels are collected across twin axes so an
  all-blank panel cannot raise a warning instead of reporting absence.

### Weight-calibration boundary

The two weights are not calibrated quantities and this delivery performs no automatic
calibration, but the **proxy used to choose them** is specified: calibrate on gradient
pressure, not on loss magnitude. The loss ratio understates the pressure — on the
recorded 100-image run the weighted Coordinate term was ~6–7% of the existing loss while
taking ~9% of the backbone gradient norm, and the weighted ranking term was ~26% of the
existing loss while taking ~8.5%. Per-objective backbone gradient norms are **not**
recoverable from the cached logged scalars, so this calibration is a deliberate,
separately authorized diagnostic pass (raw and weighted norms on a declared parameter
set, before joint clipping), never a normal-logging feature and never a runtime
auto-tuner.

The command reference publishes the measured candidate settings (Coordinate 1.0,
ranking 0.5, selected for roughly 8–10% median individual backbone gradient pressure)
together with the exact configuration and the statement that they were frozen *before*
the spatial evaluation. They are bounded candidates, not defaults, not an optimum and
not evidence of benefit; the same document records that no across-cap or
cross-recipe comparison is valid, and that a weighted dense term starting far above the
existing loss is a reason to lower the weight rather than a result. The measured cost of
that run (21 steps: baseline ~18 s versus 277/424/477 s for Coordinate/ranking/joint on
MPS) is a performance finding, not a memory-fit certificate, and it neither closes nor
replaces the outstanding production-shape memory gate.

On an entirely unavailable ranking step, keep its student head disconnected so gradients remain `None` after zeroing, avoiding artificial AdamW weight decay; ensure stale gradients cannot survive. Teacher-head EMA still follows a successful optimizer step that updates other objectives. No references or sorting network state is committed. Performance evidence should separate backbone reuse, support/geometry, projection, reference construction, sorting and backward where measurable, not present a sorter microbenchmark as end-to-end cost.

## 14. Ownership and integration

| Owner | Responsibility |
| --- | --- |
| `training/dataset.py` | Enabled-only geometry production and narrow collator; preserve historical envelopes. |
| Existing `vision.geometry` / `vision.features` | Shared checked geometry, support and differentiable sampling; no training orchestration imports. |
| Focused dense training component | Ordered pairs, eligibility/sampling, Coordinate formula and detached reduction statistics; no CLI. |
| Focused ranking component | Independent head, references and ordering loss; shared handoff works without Coordinate. |
| `training/pretrain_recipe.py` | Resolution, complete preflight, strict recovery and initialization handoff. |
| `training/pretrain_loop.py` | Reuse forwards, combine losses, preserve optimizer/EMA/FIFO order, save coordination. |
| `utils/checkpoint.py` and current consumers | Identity/architecture and initialization provenance without trainer-private imports. |
| `training/reporting.py` | Metrics and explicit recognized-header migration. |
| CLI / constraints / schema | Real options, explicitness, grouped help, no eager torch/diffsort in schema discovery. |

Do not expand `trainer.py` beyond its existing compatibility entry points. Keep one source of configuration truth and promote only demonstrated shared constants.

During implementation, synchronize master/affected design extension notes, the relevant earlier plans by dated supersession notes rather than rewriting history, bilingual README and pretrain command reference, shared augmentation/embedding-metric docs where semantics change, and `skills/otuformer-workflow/` workflow/troubleshooting references. The installed runtime schema/cards remain authoritative; skill text must not invent support on older installations. Design/plan/developer docs stay English-only. Do not advertise proposed options as available before implementation.

## 15. Verification and staged delivery

Follow the testing policy's scope/device/full-regression requirements; this section identifies affected contracts rather than creating a competing policy. Use synthetic portable fixtures; real datasets remain read-only and research runs need separate approval.

Required checks include:

- Geometry parity for all supported augmentation profiles/orientations and existing crop/resize/rotation/flip/pad/affine operations; unchanged tensor and RNG output with and without tracing; inverse/edge/padding/clipping and complete-support fixtures.
- All four batch envelopes, ragged traces, row/view alignment, explicit weak-view `None` slots without calling their geometry transforms, malformed metadata, FIFO combinations and ordinary-vs-weak teacher independence.
- Coordinate analytical values, student gradient, detached teacher, finite/nonfinite/zero norm, nearly opposed contributor cancellation at nonzero absolute norm, raw-versus-unit interpolation with unequal magnitudes, empty overlap, one point, unequal grids, exactly-once global prefix removal and already-prefix-free local tokens (including registers), and point/sample/pair reduction with intentionally unequal counts.
- Both SSL methods times four patch modes times four new-objective combinations (off/off, Coordinate only, ranking only, both); parameterize focused tests and retain representative real tiny loops, rather than declaring compatibility from mocked arithmetic alone.
- All-off tensor/RNG/loss/update/CSV/checkpoint parity, no optional import, no new parameters or transforms; ranking-only must work with Coordinate zero.
- Backbone-only teacher/student initialization, missing-selected-key failure, method switch with fresh heads, register/shape mismatch, corrupt source and identity rejection, source hash, no output mutation on preflight failure, no resume+init and no source-file requirement on later resume.
- Enabled save/resume including patch/DINO/FIFO interactions, sampling generator replay, unequal actual counts, strict config conflicts, malformed states/optimizer, never-contributed and contributed-then-skipped head state, startup dense-generator non-consumption, post-step nonfinite failure before commit, and directly affected extraction/export/CAM/finetune consumers.
- Stage timing and peak memory where measurable for baseline, Coordinate, ranking and joint modes; disclose device/dependency limitations and do not certify untested providers.

Delivery stages: (1) shared geometry handoff plus complete Coordinate/recovery, with independently schedulable backbone initialization; (2) independent NeCo chain and combinations; (3) broader contract/consumer/packaging regression and documentation acceptance. Coordinate's geometry/norm-rho qualification does not require the optional sorter; ranking additionally requires its own package/permutation qualification. Frozen recipes still require their applicable gates. Stage 2 does not wait for Coordinate scientific improvement. Full-regression triggers remain those of the central policy. Existing spatial delivery's unresolved independent re-audit and production-cost gates are not retroactively declared closed by this design.

**Acceptance boundary (author specification).** A task is accepted only with all three:
failing-then-passing named tests for the behavior it claims; measured qualification
evidence for each gate actually run on this host, with the exact code version,
configuration and device recorded; and an explicit list of gates that remain
unmeasured. Consequences: a CSV column is not reporting acceptance until a
dense-enabled run renders its diagnostics page and an all-off run still renders one
page; a preflight that accepts what the later restore rejects is not acceptance, so
the same complete check must run in both places; an in-memory replay that reuses the
live head/optimizer is not recovery evidence, because a no-op restore would pass it;
and a plotting change is not verified by reading the CSV. Document edits, hand-written
fixtures and reviewer-supplied measurements never substitute for a run in this
repository.

Research acceptance is separate: freeze split/checkpoint/layer/normalization/view protocol, compare existing objectives, plus Coordinate, plus ranking, and both. Use existing frozen same-image correspondence plus patch diversity/degeneracy and global barcode diagnostics. Loss reduction or correct geometry alone is not success; no biological gain threshold is invented here.

### Ranking-specific checks

- Exact MLP architecture, initialization, fresh head synchronization, resumed head distinction, explicit weight/bias decay groups under both existing global policies, optimizer inclusion/exclusion, EMA order and parameter-gradient reach into student backbone/head only.
- Noncommutativity fixture showing `sample(MLP(grid))` is not accidentally replaced by `MLP(sample(grid))`, and pre/post sampling normalization order.
- Pair-conditioned pools across batch rows with explicit tagged values; shared student/teacher references and ordering, allowed self/same-image refs, no cross-pair contamination, no repeated entries and no hard dependency on image identities.
- Actual counts including 0/1/2, shortages, one-image batches, unequal per-image overlap, non-power-of-two reference counts, approved reference default 16 and allowed cap 49; reject out-of-range caps before allocation, with no duplicate padding or loss of otherwise valid Coordinate points.
- Four-element permutation orientation, ascending cosine, finite gradients and teacher detachment; upstream-layout equivalence at R=4/16/49 and the factor-R distinction from an all-cell mean; identical soft-target entropy, tied/near-tied and extreme float32 cosine vectors, strict probability positivity/stochasticity, cancellation guards, and numerical failure cases.
- Reference generator replay independent from query RNG, sorter cache eviction/reconstruction parity, chunked/nonchunked gradient/reduction parity, no saved transient graphs. Exact CPU replay must cover ordered candidate-count transitions **below -> at -> above -> below** the query and reference caps (e.g. query 48/49/50/48 and reference population 15/16/17/15 for the approved sampling defaults), taking checkpoints at each boundary; compare selections, losses and subsequent generator states against uninterrupted execution with the same eligibility/pool order. Confirm no-draw branches preserve state, above-cap branches advance it, and changed eligibility is not falsely claimed to reproduce the same trajectory. Add mixed-batch fixtures: sample A has 5 eligible points while sample B has 48/49/50 before query capping, so B crosses the query cap while A remains on its no-draw branch. Separately use already-selected query counts `(5,10)`, `(5,11)`, `(5,12)` across two samples, giving reference-pool totals 15/16/17 around the reference cap16. Compare per-sample reference draws, canonical pool order, and generator states before/after each sample/batch and at restored checkpoints. The above-cap branch advances its shared reference stream and may affect later samples/batches; do not claim its effect is isolated to one sample or batch. Query/reference streams remain separate, and an identical restored eligibility sequence must reproduce the same later states.
- Ranking-only, Coordinate-only, joint and all-off behavior across existing patch/SSL combinations, ordinary teacher requirements with weak relational/FIFO settings, all-skipped head optimizer/count recovery and zero-cost optional imports.
- Full enabled save/resume and malformed-config/state/optimizer/version/library provenance preflight before outputs; read-only encoder extraction and downstream initialization/export without `diffsort`.
- Selected-device real tiny training and CPU controlled replay under the central testing policy; explicitly report untested package releases/accelerators and genuine dependency prerequisites.

### Numerical-threshold qualification before recipe freeze

The execution plan must specify and justify a reproducible **long-sequence CPU numerical/replay qualification** before freezing the provisional probability/stochasticity/norm/rho thresholds: record exact installed package/source identity, torch/dtype, seeds, supported R values, query/batch/crop coverage, iteration/query counts and planned runtime budget. Exercise representative seeded streams plus constructed equal/near-tied, extreme-cosine, unequal-magnitude and near-cancellation fixtures; replay across cap transitions and save/restart boundaries. Stream detached minima of student/teacher P, minimum feature norm and rho, and maximum absolute row/column-sum residual, with first-failure seed/case context and observed distributions/tail counts where relevant. Do not fabricate a passing margin from someone else's reported sample. Treat deliberately invalid cancellation fixtures separately from expected-valid inputs.

Report the observed margins and unresolved tails before selecting final constants; a finite replay does not prove safety for arbitrarily long training, another device or another package release. Selected-device qualification remains a separate required check. Do not widen guards merely to make the sequence pass, silently resume after a numerical failure, or change a frozen threshold within a run. If thresholds change, update recipe identity/checkpoint validation and repeat affected numerical/replay tests. This section requires a future plan gate; it does not create that plan or authorize a long run now.

## 16. Risks, unknowns, and review disposition

- Package maintenance, API/numerical behavior and supported PyTorch/device releases remain unqualified until an exact installed release is tested. Source reading and the CPU snippets below do not certify `diffsort` installation or a provider.
- Non-power-of-two sorter construction exists in the checked source, but the selected packaged release, gradients and kernels must still be qualified at actual sizes. Many small CUDA/MPS kernels may dominate sorting time; no speed or memory fit is claimed from CPU extrapolation.
- Pair-heavy local weighting, sparse actual overlap, same-image/self references, and interpolation smoothing/cancellation are known representation risks. Numerical guards diagnose numerical failure, not biological usefulness or general collapse.
- Q/R caps, dynamic reference populations and the approved reference default 16 change the reference statistics and memory demands; record actual counts and recipe provenance. Cross-cap losses/weights are not calibrated equivalents, and seed-only replay is insufficient when eligibility changes. Numerical tolerances remain candidates pending long-sequence and selected-device qualification, not analytical long-run guarantees. A persistent low-rho population triggers representation/geometry review, not automatic guard relaxation. No all-views shared reference bank or sampled-to-all-patches fallback is implied.

### Audit verification — 2026-11-15

Inline review checked official `neco.py` loss/pair slicing and exact local forward/optimizer code. A CPU in-memory AST extraction of upstream `bitonic_network` and `execute_sort` (no installation or repository helper script) verified the four-element `c @ P` fixture; upstream tensor-layout CE equals this document's `/R` formula at R=4,16,49. At R=49 a tested mismatched CE was about 6.49 (>log49), directly disproving the proposed entropy-as-CE-upper-bound claim. A follow-up using the selected unclamped CE measured the unique saved-storage values reported above and equal retained graphs under ordinary chunking. An unequal-magnitude two-vector example also distinguished raw-then-L2 from L2-then-sample targets; an almost opposed pair checked the new cancellation rule. Source SHA-256: functional `4c8ff02f74d6fe1e0b53a9685954cc25ac8f3e67ec46d7d52fdfa9f314533676`; networks `1680cf2a0e90a2d158991b91a99983ab4bb2e150feb773d6d440f4423b3b9c85`.

### Audit feedback by report identifier — received through the operator, 2026-11-15

The reviewer identifies their main series as three reports, one per round: **F1–F7/I1–I4**, **N1–N7**, and **C1–C6**. The operator's initial audit message also supplied a separate report headed `Must fix / Should fix / Nice to have`. Record that report separately below, without assigning it to the main reviewer's numbered rounds or inferring a common author. Use report identifiers rather than an uncertain receipt-to-round mapping; do not count a repeated clarification as another independent finding or audit. Reviewer tool logs, exact audited snapshots and execution independence were not independently established here. Distinguish received reports from the inline author's source checks and CPU snippets above; the author did not dispatch an independent reader/subagent.

| Report identifier | Accepted / incorporated | Not accepted as proposed, or qualified |
| --- | --- | --- |
| Main round 1: F1–F7/I1–I4 | Normalization rationale and relative cancellation; pool/query identity and decision attribution; smoothing limitation; weak-trace removal and local-prefix contract; cap bounds; explicit initialization scope, decay and package/device risks. | Row-CE mean equals **R times** the all-cell mean: using the unscaled all-cell mean as the loss instead of the row-CE mean is not equivalent. Raising the log clamp expands its zero-gradient region; use strict positive/finiteness checks instead. Separate gradient attribution is not free cached logging. Do not silently force bias exemption or remove operator-selected initialization scope. |
| Separate supplied report: Must fix / Should fix / Nice to have | Correct forward-chunking/retained-memory claims and require actual memory qualification; spell out the 6:1 pair-weight implication; provide an explicit sorting-direction example and strengthen numerical guards. These items are not attributed to the main reviewer's first report. | Retain `/R`: removing it multiplies the selected loss by R. Source verification rejected the claimed cross-pair-shared upstream batch pool. Estimated memory is not a measured GPU default peak. |
| Main round 2: N1–N7 | Default-size all-valid budget 16,960 with sorter-only 3.805/0.217 GiB extrapolations; actual-count caveats; cross-R loss/weight non-equivalence and recipe identity; provisional thresholds with long-sequence qualification; explicit 0.05 decay; 18.5%/81.5% point counts distinct from pair weights; cap-transition replay and persistent low-rho review. | Memory estimates motivate a conservative budget but do not prove **16 necessary or sufficient**. No universal nonlinear-ranking `1/sqrt(R)` law or **1.7x noise factor** was established. Threshold margins from small/reviewer samples are not analytical extreme-case or long-run guarantees; retain future qualification rather than importing them as certified evidence. |
| Main round 3: C1–C6 | Reword the all-cell-mean rejection to remove ambiguity without changing the formula; update all-round provenance; attribute the reported independent byte counts; explain constructor-inherited decay; require actual query counts beside the bound; add mixed-batch query/pool-cap replay fixtures. | The reported 4.1%/7.4% byte differences are not another measurement made here or an uncertainty interval. Feedback labelled independent by its source does not prove independently controlled execution. Shared-stream advancement may propagate beyond the current batch. |

Audits are review inputs, **not complete-design approval, plan approval, execution authorization or compatibility certification**. The reference-default reduction from 49 to16 was explicitly approved by the operator on 2026-11-15 in the follow-up correcting report attribution. That narrow approval does not approve the complete design or waive qualification gates. No CUDA/installed-package/full-training memory benchmark, long-sequence numerical qualification or scientific benefit test was performed in these document revisions.

## 17. Migration, review, and approval

This single file is the current Coordinate/ranking design. As requested by the operator, the superseded temporary record is removed rather than retained as an appendix. The selected decisions and current contracts are defined above. Related design links point here; unrelated historical delivery statuses are unchanged.

The operator approved the discussed scope, algorithms and consolidated proposals. This is a complete written **draft**, not approval of every author-specified detail or an execution authorization. Review especially proposed CLI names, strict initialization/recovery, exact state key sets to freeze during planning, dedicated RNG/device limits, per-sample references, numerical guards, cache bound, contributed-step tracking and optional dependency qualification.

Only after approval to advance from design should writing-plans produce a separate execution plan for review. One plan may stage the common handoff, Coordinate and ranking while retaining independent objective acceptance. No training code, installation, research run, commit, push or release is bundled with this migration.

**Planning handoff — 2026-11-15:** The operator subsequently requested “continue writing the plan document”, authorizing the transition to planning. The unified [implementation plan](../plans/2026-11-15-otuformer-coordinate-ranking-consistency.md) is now drafted for review. This records that later instruction without treating earlier audit/default approvals as complete-design approval, or approving plan details, implementation, dependency installation, numerical/performance runs, commits or releases.

**Audit rounds 2-5 and the current status — 2026-11-15:** Operator-supplied
re-reviews iteratively confirmed earlier fixes and found further gaps. The current status
remains **partially closed, not closed**; each round below lists what changed.

*Rounds 2-3 (fixed):* eligibility narrowed to the sampler's float32 contributor rule;
DINO accepting a legally never-contributed ranking head; one shared head-entry rule;
omitted `--model-name` inheriting the source backbone; cold replay rebuilding
head/optimizer/runtime from disk; the dense diagnostics page; strict ranking AdamW step
and moment validation including float64-overflow; a saved head checked against the frozen
`D→2048→2048→256` recipe; strict integer schema versions; a real loader length; complete
optimizer-state comparison in replay; `disabled` distinguished from `skipped`.

*Rounds 4-5 (fixed in this revision):*

- **Strict recovery is no longer FIFO-only.** An enabled dense resume runs the same
  complete check: saved optimizer, center, RNG, and both encoder states are required and
  loaded strictly, so a payload without them is rejected in preflight rather than
  resumed with a fresh optimizer or randomly initialized teacher weights.
- **Execution flags follow the real target.** The head-entry rule receives the saved
  group flags *and* the device this invocation resolves to (not the saved `args.device`),
  and the live validator rejects `capturable`/`fused` groups on a non-CUDA target.
- **Initialization sources are validated completely.** The whole selected branch
  (projector included) must match the rebuilt encoder, and every transferred tensor must
  be finite after the cast to the training dtype; a NaN projector, a wrong projector
  width, a missing projector bias or a float64 backbone value above the float32 range now
  fails at preparation.
- **`--overwrite` cannot be reached by any foreseeable failure.** A new enabled run also
  proves it has at least one full batch; a selection smaller than one batch is
  zero `drop_last` steps, not one.
- **Ranking reports its own diagnostics.** The objective returns the norm/rho minima of
  its projected teacher space, and the CSV/PDF carry them separately from Coordinate's;
  the ranking loss no longer inherits another feature space's numbers.
- **No panel is drawn without data**, including the weighted-contribution panel, and no
  "existing objectives" curve is fabricated when the CSV has no total-loss column.
- **The closed schema is strict at every level**: `ranking_config.version` must be an
  integer, dependency records reject unknown fields, and head/sort recipe fields must
  match the frozen types exactly (no `3.0` for `3`, no `1` for `True`). Resume applies the
  same inactive-option applicability rule as a new run.
- **Source verification follows the loader.** Digests are read from whatever loader
  actually imported the package, so a source-readable zip is verified instead of being
  rejected as "not a regular file"; a source-less distribution still fails with guidance.
- **The qualification runner applies production guards**: a zero student permutation
  entry is now a reported failure, and the replay covers the plan's fixed boundaries
  (query 48/49/50/48, reference pool 15/16/17/15) plus one step past the last restore.

*Round 6 (fixed in this revision):* the strict-resume RNG check now proves the snapshot
restores on the target device instead of accepting any non-empty dict; the dense-log
schema gained the 86-column predecessor so existing runs keep resuming; the fused-on-CPU
optimizer branch is accepted (only non-fused `capturable` asserts CUDA); historical
`projector.<i>.*` layouts are normalized before the strict source check; the replay
varies the real candidate count under the fixed caps (48/49/50/48 with Q49, reference
pool 15/16/17/15) and now round-trips a cold restore on the reference ladder; the
numerical runner adds a production CE backward so a detached sorter is detected; and the
ranking diagnostics report the raw pre-L2 projection norm next to the interpolated norm,
with the reference minimums and unavailable-pair count exported.

*Round 7 (fixed in this revision):* every header a previous revision actually wrote is
now a supported migration predecessor (83, 86, 88 and the current 92 columns), so a legal
earlier log never blocks its own resume; the reference-ladder cold restore now compares
the same complete state as the main replay (both generators, losses, head, full optimizer
moments, selected identity) and runs one step past its last boundary; the raw pre-L2
projection norm is aggregated per `compute` over the views that actually contributed
(`None` when none did, order-independent) instead of being overwritten per view; a payload
carrying both `projector.<i>.*` and `projector.net.<i>.*` is rejected as ambiguous instead
of silently keeping one; and the strict RNG check reuses the queue path's device policy,
so a CUDA-shaped snapshot on a non-CUDA target is structurally validated and omitted from
the restore copy, while a CUDA target demands one stream per visible device.

*Authorized repair and execution audit — 2026-11-17:* the operator authorized fixing
the remaining three findings and related execution defects. The trainer now retains the
target-filtered RNG copy before output setup and uses it at the actual restore boundary;
FIFO keeps its existing handoff, and legacy relational all-off recovery is unchanged.
Replay snapshots and compares every student/teacher head tensor, not just `0.weight`;
positive scaling of the last layer must fail even when L2 makes the loss invariant.
Teacher projection norms are cached only inside one compute and reduced over the real
queries' positive-weight bilinear contributors, excluding unused patches and empty rows.
The projection/normalization order, guards, objective values, gradients and RNG recipe
are unchanged; diagnostic support selection adds no forward or backward pass.

The related plan audit found two unfinished contracts and repaired them: fixed Q49/R16
mixed replay now holds row A at 5 while row B crosses 48/49/50/48, and uses selected
(5,10)/(5,11)/(5,12)/(5,10) reference pools with exact draw-consumption checks. Each
objective now reports global/local attempted/contributing/skipped sample/pair counts and
eligible/selected points. The enabled CSV appends these columns (124 total), preserving
native migrations from the frozen 83/86/88/92-column heads; new historical cells remain
empty. The dense PDF adds separate global/local denominator panels, staying one appended
page. Disabled-objective branch counts remain blank and the all-off header is unchanged.

Bounded CPU tests now exercise all 2 SSL × 4 patch × 4 dense combinations through one
real update, save and one resumed update, using synthetic data and offline backbone
construction. This establishes small lifecycle compatibility, not exact multi-epoch
trajectory equivalence or accelerator qualification. Separate actual-loop RNG tests cover
relational+dense, DINO+dense and DINO without dense. These are inline implementation and
verification results, not an independently dispatched review; earlier audit and benchmark
records remain historical and are not relabelled as evidence for these sources.

*Still open and explicitly not accepted:* accelerator memory, production-shape cost and
first/steady-step timing, the long (10,000-iteration) numerical stream, reference-16
memory fitness, exact full-trainer trajectory replay, multi-worker recovery, and
held-out/multiseed efficacy. The combination matrix and actual RNG handoff now have bounded
CPU coverage; this does not close the overall recipe acceptance.

**Operator-supplied follow-up review — 2026-11-17:** The author checked the supplied
nine findings against the current owners and repaired the supported findings inline.
Enabled runs now print one resolved dense JSON record per invocation with requested,
effective, applicability and requested/default/inherited sources. No all-off output or
checkpoint schema is added. Dataset items now call the existing trace `to_json`; the
collator transposes ragged records and emits canonical weak `None`; unpack calls the
existing `from_json` once per ordinary record and validates the actual view canvas.
Direct validated in-process traces remain supported at unpack, where old per-row weak
None records are normalized; the pairing interface itself accepts canonical `None` only.

Ranking guards now inspect selected student centres and positive-weight teacher support,
not unused projected patches or noncontributing rows. A pool below two entries never
enters ranking projection or its numerical guards. Raw Coordinate interpolation also
checks each positive-weight original contributor, using its existing contributor pass;
a healthy mixture cannot hide a zero contributor. Existing mixture-norm/rho guards,
FP32 computation, no-clamp row CE, sampling order and projection cache lifetime remain
unchanged. Raw-input and projection finite masks isolate unused nonfinite entries before
MLP/normalization can poison interpolation or backward; the original invalid-location
markers are retained in the norm grid and every participating invalid entry still raises.
Real-MLP backward and NaN/Inf/zero controls verify both sides. Numerical failures report
pair/row context, with row diagnosis only on the exceptional path. This tightens previously
missing raw-contributor validation and removes
the accidental whole-field ranking restriction; prior healthy-value equivalence is tested,
but earlier throughput artifacts are not re-certified against these source changes.

The three named startup-generator, contributed-then-skipped AdamW/EMA and register-prefix
controls are now real trainer tests in `test_dense_pretrain.py`. The old proposed separate
resume/reporting file references now point there; implementation checklist entries are
marked only where narrowly complete, with qualification/commit compounds left open.
Bilingual query-cap applicability now includes ranking-only. Five reported Rich failures
were independently reproduced with ANSI enabled: two assertions stripped styles too late.
The repair strips only ANSI SGR codes before whitespace normalization, preserving error,
exit-code, traceback and input/sentinel assertions; CLI production behavior is unchanged.
No independent reviewer/subagent, new research training, commit or release was invoked.

**Remaining low-risk review — 2026-11-17:** Serialized-trace codec failures now carry
geometry slot and batch-row indices while retaining the original codec message and
chained exception; missing-field, bad-matrix and bad-type regressions cover this boundary.
The plan distinguishes its seven reconciled checkboxes from the broader implemented,
bounded-tested behaviors, and labels the original all-unchecked authorization as history.
The Coordinate `dense_min_norm` aggregation is documented explicitly above and in both
command languages. Pairs with fewer than two ranking entries remain unavailable before
projection by the already documented policy; no recipe change is introduced. Current
production cost remains unmeasured: earlier throughput numbers describe earlier source
hashes, and performance-equivalence tests validate values/gradients/RNG, not throughput.
No fresh production training or long performance qualification was launched for this
review.

**Inline cross-contract audit — 2026-11-15:** After the operator-supplied E1–E3 notes, the author re-read all17 chapters, all9 plan tasks and the connected batch/geometry/model/optimizer/checkpoint/test-policy owners. The plan now closes hidden runner dependencies, source-free initialization resume, explicit weak-layout/serialized-trace validation, unavailable statistics, sampling-device conversion, support-version identity, source model fallback and contribution/skip/update tests. This is an inline source/document audit, not a dispatched independent reviewer or installed-package/numerical/memory certification.

The three-report main review series and the separate supplied report are recorded by identifier in section16 without merging their authorship or numbered rounds. The operator has separately approved reference default16; neither that decision nor the audit feedback constitutes complete-design approval. Asking whether `diffsort` is necessary was an exploratory dependency discussion, not selection or authorization of an in-repo replacement; the external optional-package proposal remains current unless explicitly revised. Builder verification performed by this author remains inline self-review and the explicitly listed source/CPU checks, not a dispatched independent reader/subagent audit. External execution provenance is limited as stated in section16. Documentation tests validate the document surfaces they cover, not algorithm correctness or future package/device compatibility.
