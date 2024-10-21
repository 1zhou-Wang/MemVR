<div align=center>
<img src="assets/memvrlogo.png" width="270px">
</div>

<h2 align="center"> <a href="https://arxiv.org/abs/2410.03577">Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models
  
</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/)
[![arXiv](https://img.shields.io/badge/Arxiv-2311.08046-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.03577)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/PKU-YuanGroup/Chat-UniVi/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2F1zhou-Wang%2FMemVR&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)      
</h5>

## 📣 News
* **[2024/10/7]** ⭐️ Paper of MemVR uploaded. Please check out [this link](https://arxiv.org/abs/2410.03577) for details.
* **[2024/11/14]**  🚀 Codes will be released on 2024-11-14. Welcome to **watch** 👀 this repository for the latest updates.

## 🎯 Overview
We propose Memory-Space Visual Retracing (MemVR), a novel hallucination mitigation paradigm without needing external knowledge retrieval or additional fine-tuning. MemVR has two significant advantages:
* First, MemVR significantly mitigates hallucination issues across various MLLMs and excels in general benchmarks, emphasizing its potential for widespread applicability.
* Second, MemVR is a plug-and-play solution without incurring added time overhead.

![MemVR](assets/compared.png)

![MemVR](assets/bigfig.png)

<div align="center">
<strong>It’s a game-changer for effectiveness and efficiency.</strong>
</div>

In contrast to previous methods, which primarily focus on eliminating biases of language priors, MemVR seeks to replenish visual clues towards more evidential responses, which signifies the other side of the coin.
Comprehensive experimental evaluations demonstrate that MemVR significantly mitigates hallucination issues across various MLLMs and excels in general benchmarks without incurring added time overhead.

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


## ✏️ Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@article{zou2024memvr,
  title={Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models}, 
  author={Xin Zou and Yizhou Wang and Yibo Yan and Sirui Huang and Kening Zheng and Junkai Chen and Chang Tang and Xuming Hu},
  journal={arxiv preprint arxiv:2410.03577},
  year={2024}
}
```
## 📝 Related Projects
- [OPERA](https://github.com/shikiw/OPERA): OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation
- [VCD](https://github.com/XiangLi1999/ContrastiveDecoding): VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
- [DoLa](https://github.com/voidism/DoLa): DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- [Contrastive Decoding](https://github.com/XiangLi1999/ContrastiveDecoding): Open-ended Text Generation as Optimization
- [GLM-4V](https://github.com/THUDM/GLM-4): ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA): Improved Baselines with Visual Instruction Tuning
- 
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=1zhou-Wang/MemVR&type=Date)](https://star-history.com/#1zhou-Wang/MemVR&Date)
