# Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models
<!-- **Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models** -->

<div style='display:flex; gap: 0.25rem; '>
  <a href="https://huggingface.co/"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"></a>
  <a href="LICENCE"><img src="assets/LICENSE-Apache%20License-blue.svg" alt="License"></a>
  <a href="https://arxiv.org/abs/2410.03577"><img src="assets/Paper-Arxiv-orange.svg" ></a>
  <a href='https://www.google.com/'><img src='https://img.shields.io/badge/Zhihu-Markdown-blue'></a>
  <a title="Hits" target="_blank" href="https://github.com/1zhou-Wang/MemVR"><img src="https://hits.b3log.org/88250/hits.svg"></a>
</div>

<div align="center">
<img src="assets/memvrlogo.png" width="25%">
</div>

## 🔥 Update
* [2024-10-7]: ⭐️ Paper of MemVR uploaded. Check out [this link](https://arxiv.org/abs/2410.03577) for details.
* [2024-11-14]: 🚀🚀 Codes will be released after one month.

## 🎯 Overview
![MemVR](assets/bigfig.png)

<div align="center">
<strong>It’s a game-changer for effectiveness and efficiency.</strong>strong>
</div>

In contrast to previous methods, which primarily focus on eliminating biases of language priors, MemVR seeks to replenish visual clues towards more evidential responses, which signifies the other side of the coin.
Comprehensive experimental evaluations demonstrate that MEMVR significantly mitigates hallucination issues across various MLLMs and excels in general benchmarks without incurring added time overhead.

## 🕹️ Usage

## 🏅 Experiments
![MemVR](assets/mmbench.png)
*Figure 5. Results on MMBench. MemVR enhances comprehensive performance on diverse tasks.*

## 📌 Examples
![Case1](assets/caseA.png)
*Figure 9. Visualization of uncertainty across layers without and with MemVR. MemVR effectively reduces uncertainty after the 8th layer, contributing to hallucination mitigations.*

![MemVR](assets/cases2.png)
*Figure 13: A case study comparing the levels of hallucination among various baselines.*

![Case2](assets/longcase.png)
*Figure 10. A case study in long text generation. MemVR effectively mitigates hallucinations.*

