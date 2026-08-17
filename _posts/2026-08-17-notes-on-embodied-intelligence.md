---
title: "Notes on Embodied Intelligence and Some Speculative Thoughts -- Compiled by GPT"
title_zh: "关于目前具身领域的认识和自己的有趣思考——由GPT整理"
date: 2026-08-17
description: "A survey of VLA, VLA+RL, and WAM, followed by a reading of JEPA through geometry, psychoanalysis, and motor learning."
permalink: /blog/notes-on-embodied-intelligence/
tags:
  - robotics
  - VLA
  - WAM
  - JEPA
---

Written late at night while waiting for experimental results.

This article reviews recent work on VLA, VLA+RL, and WAM, and then considers JEPA as another path for world modeling. The later sections read JEPA from three angles: the geometry of representation space, psychoanalysis, and motor learning. These angles do not prove one another. The mathematical discussion concerns structures that can be defined; psychoanalysis offers a language for the relation between representation and the real; neuroscience and psychology supply empirical models that can be compared.

<!--more-->

## 1. Introduction

From 2024 through the first half of 2025, VLA occupied most of the visible surface of embodied intelligence research. ICLR 2026 submissions show VLA-related papers rising from a handful to 164. Physical Intelligence's π0 and π0.5, Google's RT-2, and OpenVLA and Octo largely follow the same pattern: attach an action head to a pretrained VLM, then finetune on robot demonstrations.

From the second half of 2025, WAM papers became much more common. NVIDIA's DreamZero was the first to use the term explicitly, followed by LingBot-VA, Cosmos Policy, Motus, Fast-WAM, and GigaWorld-Policy. By mid-2026, it had become a line of work that can be compared with VLM-based VLA.

Between VLA and WAM there is also a cluster of VLA+RL papers. VLA-RL, SimpleVLA-RL, ConRFT, and π\*0.6 (RECAP) try to improve behavior-cloning policies with reinforcement learning, and they bring data efficiency, reward design, and sim-to-real into the foreground. WAM is, in part, a response to those problems.

Outside the story that video generation should carry the world model, JEPA predicts the future in a joint embedding space rather than reconstructing pixels. V-JEPA 2, reported in 2025, shows relatively high data efficiency, while leaving open questions about training stability, the action interface, and fine contact control.

Lining these papers up does not mean that each later method replaces the previous one. A more useful reading is that VLA, VLA+RL, and WAM make different choices about data, feedback, and prediction targets. JEPA changes a prior question: what should be predicted at all.

## 2. VLA: From Language to Action, and What the Mapping Drops

### 2.1 Paradigm and objective

Most VLA systems are still trained mainly by behavior cloning. Given an observation $$o_t$$ (image plus robot state) and a language instruction $$l$$, the model outputs an action sequence $$a_{t:t+H}$$:

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o,l,a)\sim\mathcal{D}} \left[ -\log p_\theta(a_{t:t+H} \mid o_t, l) \right]
$$

RT-2 (2023) showed that a VLM can be finetuned into an action generator. OpenVLA (2024) released a 7B open model trained on 970k real robot demonstrations. π0 (late 2024) used MoT and flow matching; later work such as π0.5, GR00T, and Xiaomi's robot stack adopted similar designs. Discrete action tokenizers such as FAST and BEAST are used to reduce forgetting when a VLM moves from next-token prediction to continuous action generation. Pi-FAST scored 117 points higher than a same-backbone π0-DROID on RoboArena.

<figure class="post-figure">
  <img src="{{ '/assets/img/blog/pi05.png' | relative_url }}" alt="π0.5 architecture with a VLM backbone and a smaller action expert">
  <figcaption>A typical VLA split: a pretrained VLM backbone trained with next-token prediction, and a smaller action expert trained with flow matching. Gradients from the expert do not update the backbone.</figcaption>
</figure>

### 2.2 The grounding gap: information lost when language is the goal

One difficulty for VLA can be written as a compression problem. Let the true task goal $$g^*$$ include objects, poses, contacts, and constraints, and let the language instruction $$l$$ be a compression of that information:

$$
l = \phi(g^*), \quad \dim(l) \ll \dim(g^*)
$$

VLA learns $$p(a \mid o, \phi(g^*))$$, while the desired object is closer to $$p(a \mid o, g^*)$$. The information gap between them is the grounding gap. "Put the red cup on the table" does not specify grasp pose, force, or final placement. Those variables have to be filled in from vision, scene priors, and demonstration data. More data can ease the problem, but it cannot make language carry information it never encoded.

After π0.7 added visual subgoals, action generation moved from $$p(a \mid o, l)$$ toward an inverse-dynamics problem closer to $$p(a \mid o, g_{\text{img}})$$, with a clear gain. At least on those tasks, a visual goal supplies a more specific state constraint than a short text instruction.

### 2.3 Compounding error

Behavior cloning also has compounding error. The demonstration state distribution is $$\rho_{\text{demo}}(o)$$, while the state distribution under the executed policy is $$\rho_\pi(o)$$. They need not match:

$$
\rho_\pi(o) \neq \rho_{\text{demo}}(o) \implies p_\theta(a \mid o) \text{ is not guaranteed for } o \sim \rho_\pi
$$

The mismatch accumulates over time. After a few dozen steps the policy can fail completely. Action chunking (predicting $$H$$ steps at once) reduces the problem but does not remove it, because the chunk itself remains open-loop.

## 3. VLA+RL: Introducing Trial-and-Error, and Its Limits

### 3.1 Motive and form

Behavior cloning is bounded by the demonstrator and cannot correct off-distribution drift by itself. RL is introduced to push past that limit through interaction. A typical VLA+RL objective is:

$$
\mathcal{L}_{\text{RL}} = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \gamma^t r_t \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

The KL term keeps the policy near the pretrained VLA and prevents updates from moving too fast. This is similar to the reference-policy constraint common in LLM reinforcement learning.

Representative papers include:

- **VLA-RL (2025.5)**: treats manipulation trajectories as multimodal multi-turn dialogue, optimizes an autoregressive VLA with PPO, and finetunes a VLM as a process reward model for sparse rewards.
- **SimpleVLA-RL (2025.9)**: extends the veRL stack, reports 99% on LIBERO with OpenVLA-OFT and an 80% relative gain on RoboTwin, and argues that RL can ease data scarcity while improving generalization.
- **ConRFT (2025.2)**: initializes with offline BC plus Q-learning, then finetunes an online consistency policy; gains appear after about 45 minutes of real-robot data.
- **π\*0.6 / RECAP (2025.11)**: unifies demonstration learning, error correction, and autonomous experience. The core is an advantage-conditioned policy $$\pi_\theta(a \mid o, l, A)$$. Training exposes trajectories of mixed quality; inference implicitly prefers high-advantage actions.

The empirical study "What Can RL Bring to VLA Generalization?" (2025.12) finds that PPO improves performance consistently, while DPO and GRPO are limited in partially observed settings. The POMDP character of manipulation makes simple preference optimization insufficient.

### 3.2 Bottlenecks in actual training

Existing results show that RL can keep improving a BC policy, but deployment still runs into several problems:

1. **Sim-to-real gap**. Small differences in physical parameters, friction, and sensor noise can make a simulated policy fail on a real robot. The strength of domain randomization is hard to calibrate.
2. **Reward engineering**. Sparse rewards need process reward models, and errors in those models are amplified by RL. Designing rewards on real robots is expensive.
3. **Data efficiency**. The unit cost of robot interaction (hours × number of robots × engineer time) is much higher than simulation, and the transfer value of simulated data is limited.

VLA+RL partly replaces "how many demonstrations are needed" with "how much interaction is needed," but both kinds of robot data are expensive. WAM then asks a different question: can temporal structure in video supply some physical prior first, so that the policy has less to learn from robot interaction?

## 4. WAM: Video Pretraining and the Role of "Imagination"

### 4.1 Core idea and objective

WAM's bet is that video contains dense signals about physical dynamics. A model pretrained on large-scale video should have some intuition for motion, collision, and deformation. When transferred to robot control, it mainly has to learn how actions change video, rather than learning physics from scratch.

A typical training objective is joint flow matching:

$$
\mathcal{L}_{\text{WAM}} = \underbrace{\mathbb{E}_{t,\epsilon,a} \left[ \| f_\theta(a_t, t, o, l) - (\epsilon - a) \|^2 \right]}_{\text{action prediction}} + \lambda \underbrace{\mathbb{E}_{t,\epsilon,z} \left[ \| f_\theta(z_t, t, o, l) - (\epsilon - z) \|^2 \right]}_{\text{video prediction}}
$$

Here $$z$$ is a latent representation of video frames, and $$a_t = (1-t)a + t\epsilon$$ is a noised action. The two objectives share a backbone or are coupled through MoT. Video prediction shapes physical representations; action prediction learns control.

### 4.2 Several implementations

WAM does not yet have a fixed architecture. Current systems fall into a few patterns:

- **Inverse-dynamics** (LingBot-VA, mimic-video, VPP): first generate or encode future video, then predict actions from that future representation. LingBot-VA uses a Wan 2.2-5B backbone and 16k hours of robot pretraining, reaching 74.2% on RoboTwin 2.0-Plus.
- **Joint prediction** (DreamZero, Cosmos Policy): video and action are denoised in the same model. DreamZero (Wan 14B) was the first to name WAM explicitly and scored 1750 on RoboArena. Cosmos Policy uses an action-as-image design.
- **Latent action** (Being-H0.7): compresses the future into a latent variable, trains a prior/posterior structure, and uses 200k hours of egocentric video plus 15k hours of robot demonstrations.

### 4.3 The question raised by Fast-WAM

Fast-WAM (2026.3) asks directly: is explicit future-video generation needed at test time?

Earlier WAMs followed an "imagine, then act" pattern, iteratively denoising future video at test time, with latency of 590-810ms, about 3-4 times that of π0.5. Fast-WAM keeps video co-training, but removes the future-video branch at test time and generates actions from the current frame in a single forward pass. A structured attention mask prevents action tokens from seeing future video during training, so information cannot leak.

<figure class="post-figure">
  <img src="{{ '/assets/img/blog/fast-wam.png' | relative_url }}" alt="Fast-WAM training and inference: video prediction is kept as a training objective, while test-time action generation does not attend to future video">
  <figcaption>Training can still use video denoising. At test time, the cheaper option is to predict actions without generating future video, using video prediction only as a representation-learning objective.</figcaption>
</figure>

Controlled results:

| Method | RoboTwin | LIBERO | Latency |
|---|---|---|---|
| Fast-WAM (no test-time imagination) | 91.8% | 97.6% | 190ms |
| Fast-WAM-Joint (joint imagination) | 90.6% | 98.5% | ~590ms |
| Fast-WAM-IDM (imagine, then act) | 91.3% | 98.0% | 810ms |
| Video co-training removed | 83.8% | 93.5% | 190ms |

In these experiments, **explicit future generation at test time contributes little, while video prediction during training matters much more.** On a real towel-folding task, removing video co-training dropped success to 10%, while the three test-time imagination modes differed only slightly.

A plain reading is that most of WAM's gain happens in training. The video-prediction objective improves temporal representations; whether a future video is actually generated at test time matters less. Later work such as Faster-WAM and GigaWorld-Policy-0.5 also moves toward "predict in training, act directly at test time."

### 4.4 Limits of WAM

Video pretraining does give robot policies useful temporal signal. At least two issues still need to be separated:

First, **the surface character of video physics**. Video generators learn pixel statistics that look plausible, not mechanical causation. Object interpenetration and discontinuous motion are common in generated video, so policies based on such world models remain uncertain. The gain from video co-training in Fast-WAM may come from better visual-temporal features rather than genuine physical understanding.

Second, **a blurred boundary with VLA**. If imagination is unnecessary at test time, the difference between WAM and "a VLA with a video-model backbone" may reduce to the pretraining objective (video prediction versus image-text matching). How large that difference is still needs controlled comparison.

## 5. JEPA: Latent Prediction as Another World Model

<figure class="post-figure post-figure-portrait">
  <img src="{{ '/assets/img/blog/yann-lecun.png' | relative_url }}" alt="Sketch of Yann LeCun at a desk with world-model notes">
  <figcaption>Yann LeCun, whose JEPA papers argue for predicting in representation space rather than reconstructing pixels.</figcaption>
</figure>

### 5.1 Core idea and mathematical form

JEPA starts from a simple claim: **predict representations, not pixels.**

Its energy function is:

$$
F_\theta(x, y) = \min_z \| s_y - g_\theta(s_x, z) \|^2
$$

Here $$s_x = h_\theta(x)$$ is the context representation, $$s_y = h_{\theta'}(y)$$ is the target representation ($$\theta'$$ is an EMA of $$\theta$$, with stop-gradient), $$g_\theta$$ is the predictor, and $$z$$ is a latent that encodes target location or masking. The training objective is:

$$
\mathcal{L}_{\text{JEPA}} = \mathbb{E}_{(x,y)} \left[ F_\theta(x, y) \right]
$$

Unlike generative WAM, $$s_y$$ is not constrained by pixel reconstruction. The model does not need to render the future; it only needs to predict a target representation. In the ideal case, the encoder keeps changes related to time and action, and suppresses background texture, lighting, and sensor noise. What is actually kept still depends on the data, the masking policy, and the anti-collapse mechanism.

<figure class="post-figure">
  <img src="{{ '/assets/img/blog/vjepa.png' | relative_url }}" alt="V-JEPA architecture with a context encoder, an EMA target encoder, a predictor, and a stop-gradient loss">
  <figcaption>V-JEPA predicts masked latent tokens from a context encoder. The target encoder is an EMA copy and is blocked by stop-gradient, so the loss compares representations rather than pixels.</figcaption>
</figure>

### 5.2 Geometry of latent space: state manifolds and quotient spaces

Pixel observations live in a high-dimensional space $$\mathcal{X}\subset\mathbb{R}^{H\times W\times C}$$, but robot tasks usually involve far fewer degrees of freedom. Object poses, joint states, contact relations, and material deformation form a lower-dimensional state set that can be treated, approximately, as a manifold $$\mathcal{M}$$. The encoder's job can be written as:

$$
h:\mathcal{X}\rightarrow\mathcal{M}\subset\mathbb{R}^d
$$

The same physical state can appear under different lighting, textures, and viewpoints. If a group $$G$$ stands for those task-irrelevant transformations, the desired object is closer to a quotient space $$\mathcal{X}/G$$: observations that look different but are physically equivalent should map to nearby representations. "Manifold" here is a modeling assumption, not a property that appears automatically once JEPA is used. If data coverage is thin, the latent space can bend, tear, or compress away contact information that control still needs.

<figure class="post-figure">
  <img src="{{ '/assets/img/blog/manifold.png' | relative_url }}" alt="High-dimensional functions encoded onto a lower-dimensional manifold and mapped in latent space">
  <figcaption>A schematic of compressing high-dimensional observations onto a lower-dimensional manifold. The figure includes a decoder; JEPA keeps the latent path and drops pixel reconstruction.</figcaption>
</figure>

An action $$a_t$$ then induces local dynamics on the manifold:

$$
s_{t+1}=T_{a_t}(s_t), \qquad T_{a_t}:\mathcal{M}\rightarrow\mathcal{M}
$$

In continuous time, an action can also be read as choosing a vector field,

$$
\dot{s}=f(s,a).
$$

This rewriting changes JEPA's target from "generate the next frame" to "estimate how the state moves along feasible actions." It also gives a clearer failure criterion. If the encoder maps two dynamically different states to the same point, the predictor cannot emit the correct successor for both. If the representation keeps too many appearance degrees of freedom, the local dynamics become unnecessarily complicated.

Locally, the encoder Jacobian $$D h_x$$ describes how observation changes are compressed into latent space. If it drops rank along task-relevant directions, those directions are wrongly merged. Total collapse is the extreme case $$\operatorname{rank}(D h_x)=0$$. Full rank is not enough either, because irrelevant directions such as lighting and texture may still occupy many dimensions. Anti-collapse should not merely keep variance nonzero; it should keep the right local degrees of freedom.

The map $$T_{a_t}$$ also hides a Markov assumption. A single image usually cannot determine velocity, occluded state, or contact history, so the same $$s_t$$ may have several successors. A better state should be formed from observation history, for example $$s_t=h(o_{\leq t})$$; or $$\mathcal{M}$$ should be read as a belief manifold over the true state. Otherwise the predictor faces a multi-valued flow rather than a single trajectory.

Latent space also needs a suitable geometry. Euclidean distance $$\lVert s_i-s_j\rVert_2$$ treats every direction as equally costly, but distances in control are often directional: pushing a cup off a table is easy, restoring it is not. A more natural object is a quasimetric $$d(s_i,s_j)$$ induced by reachability or cost-to-go, which need not be symmetric. That is why the quasimetric results in Intrinsic-Energy JEPA are worth attention: energy need not only measure similarity; it may encode how hard it is to go from one state to another.

### 5.3 Differences from WAM

WAM's flow matching models a conditional distribution $$p(y \mid x)$$ and therefore has to describe probability mass over an output space. JEPA's energy function only has to compare candidate targets; it need not reconstruct a full output. For control, the latter can be more direct, because a policy often cares which futures are reachable and cheaper, not how every pixel is generated.

Another difference is the reconstruction constraint. Even when WAM works in a VAE latent space, the information must still be rich enough to decode back to video. JEPA has no such requirement, so it can discard more appearance detail. The cost is equally direct: whether the discarded information was truly irrelevant to control cannot be judged by looking at a generated video.

WAM's videos can be inspected by eye. JEPA's latent space can be assessed only through downstream tasks, linear probes, or geometric diagnostics. This is not a simple tradeoff between interpretability and efficiency. A visualized video may only show surface coherence, while an undecodable latent space can still be studied through neighborhoods, reachability, and intervention experiments.

### 5.4 Experimental evidence from V-JEPA 2

V-JEPA 2 (2025.6) is currently the strongest empirical check of the JEPA line. It has 1.2B parameters and is pretrained with self-supervision on 1 million hours of internet video plus 1 million images. Its action-conditioned variant, V-JEPA 2-AC, is post-trained on **fewer than 62 hours of unlabeled robot video** from DROID, then zero-shot controls a Franka arm on pick-and-place in an unseen lab, with 65-80% success.

That data scale contrasts with common VLA setups: OpenVLA uses 970k labeled demonstrations, and π0.5 uses heterogeneous multi-robot data plus web data. That fewer than 62 hours of unlabeled robot video can support zero-shot control suggests that the pretrained representation contains transferable temporal structure. It does not show that the model has acquired a general physical representation.

V-JEPA 2 also has a clear limit. Reproduction experiments find that the arm can approach objects reliably, but often fails at the final contact or grasp: the end-effector hovers above the object and the gripper does not close. This suggests that JEPA has learned high-level motion planning (where to go), but not fine contact control (how to grasp). The latter needs touch, force, and proprioception, which purely visual video does not provide.

### 5.5 Engineering bottlenecks

Current issues for JEPA include:

1. **Unstable training**. Joint-embedding architectures collapse easily when the predictor is too strong or the target encoder is not updated, so all representations become similar. I-JEPA uses an EMA target encoder, multi-block masking, and regularization; V-JEPA 2 adds more engineering tricks. The overall pipeline is more complex than contrastive learning or generative pretraining.
2. **Latent space is hard to diagnose**. Predicted content cannot be inspected directly. Iteration depends on downstream tasks, so the feedback loop is long.
3. **Immature action-conditioning interface**. V-JEPA 2-AC injects actions in a separate post-training stage, using pseudo-labels from inverse-dynamics estimation. There is no standard answer yet for action encoding, multi-step planning, or combination with RL.
4. **Viewpoint bias in the data**. Internet video is mostly third-person, human-centered, and edited. Robot control needs first-person, object-centered, interaction-dense video.

Some of these are implementation problems. Some may reflect information limits of the objective itself. They are still hard to separate cleanly.

## 6. Lacan's Three Registers as a Reading Frame

Lacan's Real, Symbolic, and Imaginary can be used as a reading frame. The point is not that models possess a subject, and not that psychoanalysis explains an optimizer. The question is narrower: when a model approaches the physical world through language, images, or latent representations, what does it establish, and what does it leave out?

<figure class="post-figure">
  <img src="{{ '/assets/img/blog/lacan-rsi.png' | relative_url }}" alt="Triangular diagram of Lacan's Real, Symbolic, and Imaginary">
  <figcaption>The three registers as a reading diagram, not as a taxonomy of models. The extra symbols mark relations among the registers; they are not mapped onto VLA, WAM, or JEPA.</figcaption>
</figure>

### 6.1 VLA and the Symbolic: an instruction cannot close a task

Language instructions belong to a symbolic order. Through difference and rule they organize experience, so that "pick up the cup" and "put it on the table" become tasks that can be communicated, reused, and composed. Symbolization is never a complete copy of a physical process. The same instruction can correspond to many grasp trajectories, and the same trajectory can be described in different words.

"The big Other is barred" does not supply a new loss here. It is a reminder that language is incomplete. An instruction can constrain a task, but it cannot underwrite every contact event, force, and pose. VLA still has to fill in context from vision and demonstrations, and that filling-in changes with the scene.

Objet petit a should also not be identified with "a few missing physical variables in the sentence." A more careful claim is that a task description always leaves a remainder that description does not exhaust. More data, prompts, and state inputs can shrink that remainder. They are unlikely to produce an instruction that is closed for every scene.

That is why the visual subgoals in π0.7 are interesting. They do not erase the distance between language and action. They add a different kind of constraint: language says what to do, and an image shows one possible completed state. The two representations do not cover the same content.

### 6.2 WAM and the Imaginary: the mirror, misrecognition, and the ideal ego

WAM sits easily next to the Imaginary because it organizes the future as a continuous, visible scene. The point of the mirror stage is not that vision is false. It is that the unity of a complete image can precede, and even conceal, the incoherence of bodily experience.

Generated video can have a similar effect. A folding or grasping sequence that looks coherent is easily taken as evidence that "the model understands physics." Visual coherence and dynamical correctness are not the same thing. Objects may interpenetrate locally, contact forces may never be represented, and errors may be hidden by smooth texture and motion.

Fast-WAM unsettles this reliance on a complete picture. After test-time video generation is removed, performance does not disappear with it. The visible "imagining" is therefore not necessarily a required step of the policy. In the language of the Imaginary, generated video is more like a viewable image of the model's capacity. It has diagnostic value, but it should not automatically be taken as the internal inference itself.

### 6.3 JEPA and the Real: invisibility is not closeness to truth

The Real, in Lacan's sense, is not a synonym for objective reality. It names what a given symbolic order cannot fully absorb. JEPA does not require latent representations to become pixels again, and therefore does not promise a complete, viewable future. That lets it avoid WAM's pursuit of visual completeness, and it leaves a kind of invisibility inside the representation.

Invisibility is not the same as being closer to the real. An undecodable latent space may preserve contact structure, or it may simply have lost information. It may form a useful abstraction, or it may collapse. Psychoanalysis can warn against equating visualization with understanding. It cannot replace ablations, control performance, or geometric diagnostics.

From this angle, the more interesting feature of JEPA is that it admits representation must choose. The model does not try to carry every visible detail into the future; it decides, under a prediction task, which differences are worth keeping. Something always falls outside the representation: touch, force, physical states not covered by the data, or changes the objective treats as irrelevant. These remainders should not simply be named objet petit a, but they reappear when the model fails.

This also changes what "world model" can mean. A world model is no longer an unfoldable replica of the world. It is a set of differences in the service of action. If two frames differ a great deal in pixels but afford the same action, the representation can place them close together. If two frames look almost identical but sit on different sides of a contact event, the representation should separate them. The criterion comes from action consequences, not from whether an image looks realistic.

### 6.4 The three registers are not a taxonomy of models

Assigning VLA to the Symbolic, WAM to the Imaginary, and JEPA to the Real can only be a sketch of tendencies, not a strict classification. VLA still depends on visual representations, WAM still takes language as a condition, and JEPA's latent space is still a symbolic system built from data and an objective.

The Borromean knot is more useful as a picture of three demands holding one another in place. Language makes a task sayable, images make a state recognizable, and collision, friction, and contact keep exposing what the first two cannot cover. A deployable system has to keep the three together. The visual subgoals in π0.7, Motus's hybrid architecture, and VLA-JEPA can be read as different ways of making a join. Whether they form a stable knot is still a question for real interaction.

<figure class="post-figure">
  <img src="{{ '/assets/img/blog/borromean.png' | relative_url }}" alt="Three interlocking Borromean rings">
  <figcaption>If any one ring is removed, the other two fall apart. The image is topological, not a claim that VLA, WAM, and JEPA occupy the three colors.</figcaption>
</figure>

## 7. Neuroscience and Psychology: How Prediction Enters Action

### 7.1 Forward models, inverse models, and efference copies

Internal-model theory in motor neuroscience offers a reference for these lines of work. Motor control is often split into two computations:

- **Forward model**: $$\hat{s}_{t+1} = f_{\text{forward}}(s_t, a_t)$$, predicting the sensory consequences of an action, updated through an efference copy and prediction error, and compensating for 100-200ms of sensory delay.
- **Inverse model**: $$a_t = f_{\text{inverse}}(s_t, s_{t+1}^*)$$, computing a motor command from a target state.

If one compares computational roles only, an incomplete correspondence looks like this:

| Neuroscience | VLA | WAM | JEPA |
|---|---|---|---|
| Inverse model | action head | action expert | (interface incomplete) |
| Forward model | weak or implicit | pixel-level video prediction | representation-level latent prediction |
| Efference copy | usually not modeled explicitly | action-conditioned video | action-conditioned latent prediction |
| Prediction error | mainly action supervision | video prediction error | representation prediction error |

The table describes functional similarity, not biological homology. A nervous system does not render a high-resolution future image before choosing an action, so representation-level prediction sits more comfortably with internal-model theory. That does not imply that JEPA is "more like a brain." Biological forward models are constrained by body structure, timescales, and multiple sensory channels, while current JEPA systems learn mainly from visual statistics.

The efference copy matters in particular. When a hand reaches out, the nervous system does not only receive the next visual frame. It also uses a copy of the motor command to predict changes in proprioception and touch, and thereby distinguishes self-generated change from external disturbance. Action-conditioned JEPA has a similar computational interface, but if the training signal is only video, it still cannot tell whether a visually correct approach came with a stable grasp force.

### 7.2 Affordance and task-relevant representation

Affordance, in ecological psychology, concerns what the environment offers an actor, not only what category an object belongs to. The same cup handle is graspable in different ways for a human hand, a two-finger gripper, and a suction cup. Whether a surface is "placeable" depends on object size, friction, and the state of the end-effector.

This view connects naturally to JEPA's latent space. A representation useful for control need not keep every pixel of a cup, but it should keep edges, poses, occlusions, and contact relations that bear on graspability. Further, a state representation should depend not only on current observation, but also on observation history $$o_{\leq t}$$ and on the actor's body and capacities $$b$$:

$$
s_t = h(o_{\leq t},b).
$$

If $$b$$ is ignored, a so-called task-irrelevant abstraction may simply have deleted the relevant bodily constraints. That V-JEPA 2-AC works in the approach phase and fails at contact is at least compatible with this problem: a visual direction toward the target has formed, while the body-object relation needed for grasping is still missing.

### 7.3 These lines of work seen through stages of learning

Fitts and Posner divide motor-skill learning into cognitive, associative, and autonomous stages. Beginners rely on verbal rules and visual monitoring. With practice, actions and sensory consequences form more stable links. Once skilled, control becomes fast and no longer needs step-by-step explicit reasoning.

There is a loose functional analogy with VLA, VLA+RL, and Fast-WAM: language and demonstration supply an initial structure, trial-and-error corrects action-outcome relations, and after enough training the policy executes directly. This is not a one-to-one map from three algorithms to three psychological stages. A mature system still uses instruction, prediction, and feedback together, and humans also fall back from autonomous control to conscious adjustment in new situations.

What is harder to avoid is the body. Human motor learning uses proprioception, touch, vision, vestibular signals, and pain, and it actively changes movement in order to obtain more informative sensation. A purely visual world model can see whether an object moved, but not how much normal force the gripper applied. It can observe the shape of cloth without knowing the tension in the material. For JEPA, the next step is not only to scale video data. It is to decide how different sensory modalities enter the same state space, and how prediction error should receive higher weight when contact occurs.

## 8. A Combined Assessment

**VLA**. The information lost in a language goal will not disappear merely by scaling data. Visual subgoals, state constraints, and execution feedback are likely to become standard components. VLA remains useful in structured scenes, but current evidence is not enough to support it as a standalone solution for long-horizon operation in open environments.

**VLA+RL**. Current RL is used more as a refinement after BC than as a way of discovering skills from scratch. RECAP is a reasonable direction in putting demonstration, correction, and autonomous experience into one pipeline. The limits remain interaction cost, reward reliability, and sim-to-real. In the near term this line will depend on substantial hardware and engineering.

**WAM**. Video pretraining contributes a clear temporal signal, but "explicit future imagination" may have been overemphasized. Fast-WAM shows that, at least on current benchmarks, generating video at test time is not necessary for the gain. The boundary between WAM and VLA may therefore show up more as a difference in pretraining objective and backbone choice.

**JEPA**. Representation-level prediction reduces the burden of pixel reconstruction and leaves room for state manifolds, reachability, and asymmetric control cost. V-JEPA 2 shows that this path can learn transferable structure, but failures at contact also show that abstraction does not always keep the information control needs. Training stability, latent-space diagnosis, action conditioning, and multimodal feedback still lack mature answers.

**Shared problems**. Robot data is expensive, continual learning after deployment remains weak, and perception is still concentrated on vision. How touch, proprioception, and the acting body itself enter the representation may matter more than the next renamed backbone.

Hybrids are already appearing. π0.7 uses BAGEL to generate visual subgoals, then an action expert executes them. Motus and BagelVLA jointly train understanding, video generation, and action. VLA-JEPA begins to attach latent prediction to a policy. Rather than waiting for one acronym to win, the more practical questions are how these modules share state, how they use real feedback, and whether the system can keep learning after it fails.

---

> **Representative work**
>
> VLA: RT-2, Octo, OpenVLA, π0, π0.5, π0.7, GR-2, X-VLA, FAST, BEAST
>
> VLA+RL: VLA-RL, SimpleVLA-RL, ConRFT, CO-RFT, π\*0.6/RECAP, ReinboT, VLA-RFT, STARE-VLA
>
> WAM: UniPi, GR-1, DreamZero, LingBot-VA, Cosmos Policy, Motus, Fast-WAM, Faster-WAM, GigaWorld-Policy, GigaWorld-Policy-0.5, Being-H0.7, SelfWAM, AHA-WAM
>
> JEPA: I-JEPA, V-JEPA, V-JEPA 2, V-JEPA 2-AC, VLA-JEPA, Intrinsic-Energy JEPA, Var-JEPA
>
> Theoretical references: Wolpert & Kawato (internal models), Gibson (affordance), Fitts & Posner (stages of motor learning), Lacan (Real / Symbolic / Imaginary)
