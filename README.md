# LUMINA: Detecting Hallucinations in RAG System with Context–Knowledge Signals (ICLR 2026)

By [Samuel Yeh](https://mhyeh.github.io/), [Sharon Li](https://pages.cs.wisc.edu/~sharonli/index.html), and [Tanwi Mallick](https://www.anl.gov/profile/tanwi-mallick).

[![Paper](https://img.shields.io/badge/arXiv-2509.21875-orange)](https://arxiv.org/abs/2509.21875)

## Overview

LUMINA is a novel framework that detects hallucinations in RAG systems through context-knowledge signals. The key insight is that hallucinations in RAG often stem from an imbalance between how models use external context and their internal knowledge. LUMINA quantifies these two signals:

- **External Context Utilization**: Measured via distributional distance between predictions conditioned on relevant vs. random documents
- **Internal Knowledge Utilization**: Measured by tracking how predicted tokens evolve across transformer layers

The method is layer-agnostic and requires minimal hyperparameter tuning, making it more generalizable than prior approaches. LUMINA achieves consistently high AUROC and AUPRC scores, outperforming prior utilization-based methods by up to +13% AUROC on HalluRAG .


## Usage

```python
from lumina import LUMINA
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('...')
tokenizer = AutoTokenizer.from_pretrained('...')

detector = LUMINA(model, tokenizer)

prompt_w_context = "Instruction: [INSTRUCTION] Context: [CONTEXT]"
prompt_w_random_context = "Instruction: [INSTRUCTION] Context: [RANDOM CONTEXT]"
response = "[RESPONSE]"

# Returns (hallucination_score, mmd, ipr)
hallucination_score, mmd, ipr = detector.predict(
    prompt_w_context, 
    prompt_w_random_context, 
    response
)

# Higher hallucination_score indicates higher likelihood of hallucination
# You can also access individual components:
# - mmd: External context score (higher = better context utilization)
# - ipr: Internal knowledge score (higher = more internal knowledge reliance)
```

The `predict` function returns three values:
- **`hallucination_score`**: Combined score where higher values indicate higher hallucination likelihood
- **`mmd`**: External context utilization score (from MMD computation)
- **`ipr`**: Internal knowledge utilization score (from IPR computation)

## A Quick Walkthrough on LUMINA

**External Context Score**

The external context score measures how sensitive the LLM is to semantic changes in the input documents. The core idea: if the model effectively uses external context, replacing relevant documents with random ones should significantly change the token probability distribution.

This is quantified using **Maximum Mean Discrepancy (MMD)** [1], a kernel-based statistical distance measure between two probability distributions:


```python
def __compute_mmd(self, p_prob, q_prob, embedding_layer, k=100, **kernel_kwargs):
    p_top_k_list, p_embed_list = self.__get_topk_embeddings_and_probs(p_prob, k, embedding_layer)
    q_top_k_list, q_embed_list = self.__get_topk_embeddings_and_probs(q_prob, k, embedding_layer)
    with torch.no_grad():
        K_pp = torch.stack([p_top_k @ self.kernel(p_embed, p_embed, **kernel_kwargs) @ p_top_k.T for p_top_k, p_embed in zip(p_top_k_list, p_embed_list)])
        K_qq = torch.stack([q_top_k @ self.kernel(q_embed, q_embed, **kernel_kwargs) @ q_top_k.T for q_top_k, q_embed in zip(q_top_k_list, q_embed_list)])
        K_pq = torch.stack([p_top_k @ self.kernel(p_embed, q_embed, **kernel_kwargs) @ q_top_k.T for p_top_k, p_embed, q_top_k, q_embed in zip(p_top_k_list, p_embed_list, q_top_k_list, q_embed_list)])
    return (K_pp + K_qq - 2 * K_pq).cpu()
```

- `p_prob`: Token probabilities when the model sees the **correct retrieved documents**
- `q_prob`: Token probabilities when the model sees **random documents**
- For each distribution, we extract the top-k most probable tokens and their embeddings
- We compute kernel similarities within and between distributions
- **Higher MMD** = larger distributional difference = model is more sensitive to context changes = **higher external context utilization**


**Internal Knowledge Score**

The internal knowledge score tracks how the model's predictions evolve across transformer layers using a mechanistic interpretability tool called **Logit Lens** [2]:


```python
logit_lens_res = []
with torch.no_grad():
    for l, hid in enumerate(answer_hid_w_context):
        if hasattr(self.model.model, 'language_model'):
            lens_logits = self.model.lm_head(self.model.model.language_model.norm(hid)).float()
        else:
            lens_logits = self.model.lm_head(self.model.model.norm(hid)).float()
        logit_lens_res.append(F.softmax(lens_logits, dim=-1))
```

- For each transformer layer, we take the hidden states and project them into vocabulary space
- This reveals what the model "thinks" the next token should be at each layer
- We store the probability distribution over tokens at each layer

Then we compute the **Information Processing Rate (IPR)**:

```python
def __compute_ipr(self, hid_prob, ans_prob, ans_ids):
    T = ans_prob.shape[0]
    ipr = []
    for t in range(T):
        layer_ratio = []
        max_id = torch.argmax(ans_prob[t]).to("cuda")
        total_weight = 0
        for l in range(len(hid_prob)):
            entropy = self.__compute_entropy(hid_prob[l][t])
            w = 1.0 / (entropy.item() + 1e-8)
            l_index = l + 1
            ratio = 1 - min(hid_prob[l][t][max_id].item() / ans_prob[t][max_id].item(), 1.0)
            layer_ratio.append(ratio * l_index)
            total_weight += l_index * w
        ipr.append(sum(layer_ratio) / total_weight * ans_prob[t][ans_ids[t]].item() / ans_prob[t][max_id].item())
    return torch.tensor(ipr)
```

- For each token in the generated answer, we compare predictions at each intermediate layer to the final output layer
- If the model's prediction doesn't converge until later layers, it suggests the model is adding more information during processing (likely from internal knowledge)
- We weight deeper layers more heavily (multiplied by `l_index`)
- We weight layers with lower entropy (more confident predictions) more heavily
- The final IPR score is higher when:
  - Early layer predictions differ significantly from the final prediction
  - The model relies more on internal processing rather than just copying from context
- **Higher IPR** = more internal knowledge utilization = potential over-reliance on parametric knowledge

### Combining the Scores

LUMINA combines both scores using a weighted linear combination to produce the final hallucination score:
```python
hallucination_score = λ × IPR - (1 - λ) × MMD
```

- **λ (lambda)**: A hyperparameter that balances the contribution of internal knowledge vs. external context signals (set via `self.lam`, default is `0.5`)
- **IPR (Internal Knowledge Score)**: Higher values indicate greater reliance on parametric knowledge
- **MMD (External Context Score)**: Higher values indicate stronger context utilization

The formula captures the key insight from the paper that hallucinations occur when there's an **imbalance** between internal and external signals.

## References

[1]: Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(25):723–773, 2012. ISSN 1533-7928.

[2]: nostalgebraist. interpreting gpt: the logit lens, 2020. URL https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

---

## Citation
```
@inproceedings{yeh2026lumina,
  title={LUMINA: Detecting Hallucinations in RAG System with Context–Knowledge Signals},
  author={Samuel Yeh and Sharon Li and Tanwi Mallick},
  booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
}
```

## License
This work is released under the MIT License.