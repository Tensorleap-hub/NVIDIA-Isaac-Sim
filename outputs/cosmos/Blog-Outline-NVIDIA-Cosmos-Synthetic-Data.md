# Blog outline \- Tensorleap x NVIDIA, synthetic data 

**Reference:** [Laguna Health \- How Laguna Health Cut Clinical AI Inference Costs by 85% with NVIDIA Nemotron](https://www.lagunahealth.com/blog/how-laguna-health-cut-clinical-ai-inference-costs-by-85-with-nvidia-nemotron)

**Working title:** Synthetic Data That Closes the Sim-to-Real Gap: Tensorleap with NVIDIA Isaac Sim and Cosmos **Subhead:** Generating more synthetic data isn't the bottleneck. Generating the right data is. **Byline:** Or and Yotam

---

## 1\. The problem (\~250 words)

Physical AI models fail in the field in ways that never show up in validation. Warehouse perception is the concrete case: a detector trained on available data meets a camera at a different height, a different field of view, different clutter and motion.

Synthetic data is the standard answer, and the tooling is now excellent. But teams generate more frames and get diminishing returns, because nothing tells them whether the frames they generated match the gap they actually have. Visual realism is not the test. A photorealistic frame of the wrong scene is still the wrong scene.

Land the reframe here: this is a targeting problem, and the model's latent space is the only honest judge.

## 2\. Why the latent space is the measuring instrument (\~200 words)

Short, and the most differentiated section in the piece. The detector's latent space is where the model's own view of the data lives. Two datasets that look similar to a human can sit far apart there, and that distance is what predicts whether training on one helps on the other.

This is Tensorleap's core capability doing the work: not a new idea invented for this project, but the same latent-space analysis applied to a generation problem.

## 3\. The pipeline (\~300 words \+ existing diagram)

Walk the loop concretely, naming every NVIDIA component precisely:

1. Render warehouse scenes in **NVIDIA Isaac Sim / Isaac SDG** with parameters θ  
2. Embed the synthetic samples through the trained **RF-DETR** detector  
3. Measure distance to the real-data distribution in that latent space, **MMD-RBF objective**  
4. **Tensorleap's optimization pipeline** proposes the next θ  
5. Loop until the distance converges  
6. Hand the converged θ to **Cosmos-Transfer2.5**, NVIDIA's generative world model, for a photoreal video-to-video pass with depth, edge and segmentation control  
7. Fine-tune the detector on the photoreal calibrated data

Note that Isaac SDG and Cosmos-Transfer2.5 are two applications on the same Omniverse platform \- the whole loop runs on one stack.

## 4\. Results (\~350 words \+ existing chart)

Lead with the table. Be explicit about the baseline.

| Training mix | F1 on real LOCO images |
| :---- | :---- |
| Real only | 0.476 |
| Base synthetic | 0.582 |
| Base synthetic \+ Cosmos | 0.622 |
| Calibrated (Opt-0) | 0.638 |
| **Calibrated \+ Cosmos** | **0.688** |

Three readings, in order of interest:

- **Calibration alone beats generation alone.** 0.638 vs 0.622. Getting the geometry right matters more than getting the appearance right.  
- **They compound.** Against base synthetic, Cosmos adds 0.040 and calibration adds 0.056, but together they add 0.106 \- slightly more than the sum. They are closing different gaps. State this carefully: single run, small margin, directionally clear rather than proven.  
- **The headline number.** \+0.106 F1 over base synthetic, or \+0.212 over real-only training. Name the baseline explicitly.

Include the per-epoch curves for mAP@50, mAP@50-95 and AP pallet\_truck.

## 5\. What calibration actually changed (\~250 words \+ existing gallery)

The before/after gallery is the most persuasive asset. Walk the four failure modes as evidence that the loop found real problems:

- Extremely cluttered, no navigable aisle → human-height camera, clear aisles  
- Low camera, narrow FOV → ground-truthed height, wide angle  
- Implausible object placement → realistic motion with blur  
- Security-camera overhead angle → elevated but legible

The point: nobody hand-specified these fixes. The loop found them by minimizing latent distance. And each un-calibrated scene, run through Cosmos, produces a beautiful render of the wrong thing.

## 6\. Limitations (\~150 words)

Do not skip. This is what makes the results credible rather than promotional.

Be straight about: one vertical (warehouse), one detector architecture (RF-DETR), one real dataset (LOCO subset-3), a single optimization round (Opt-0), single training runs rather than seed-averaged, and no cost or wall-clock comparison yet. Say what would strengthen it: more rounds, more verticals, repeated runs.

## 7\. What this means for physical AI teams (\~200 words)

Close on the reliability layer. The loop here is find, fix, verify \- the same loop Tensorleap runs on production models, pointed at the data-generation stage. As simulation makes data abundant, the differentiator is no longer who can generate the most, but who can tell what to generate and prove it worked.

One sentence positioning Tensorleap alongside NVIDIA rather than on top of it: NVIDIA built the generation stack; this work is about aiming it.

Name NVIDIA Inception membership here.

## 8\. Method footnote

Italic, at the end. Dataset, detector, control signals, evaluation protocol, what "Opt-0" denotes, clip playback speed.

---

## Open questions before drafting

1. **Which baseline do we headline?** \+0.106 over base synthetic is the honest apples-to-apples claim. \+0.212 over real-only is bigger but answers a different question. Recommend leading with the first and mentioning the second.  
2. **What does "Opt-0" mean?** It appears in every chart and is never defined. If it is the first optimization round, saying so strengthens the story \- there is more headroom.  
3. **Any cost or time figures?** GPU hours, wall-clock to convergence, or frames needed versus the base approach. Even a rough figure would lift the piece.

