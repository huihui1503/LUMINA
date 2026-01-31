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
@torch.no_grad()
def __compute_mmd(self, p_prob, q_prob, embedding_layer, k=100, **kernel_kwargs):
    """
    Compute Maximum Mean Discrepancy using vectorized operations.
    """
    # Get top-k for both distributions
    p_top_k, p_embed = self.__get_topk_embeddings_and_probs(p_prob, k, embedding_layer)
    q_top_k, q_embed = self.__get_topk_embeddings_and_probs(q_prob, k, embedding_layer)
    
    T = p_prob.shape[0]
    mmd_scores = []
    
    # Process in batches to avoid memory issues with very long sequences
    batch_size = 32
    for i in range(0, T, batch_size):
        end_idx = min(i + batch_size, T)
        
        # Compute kernel matrices for batch
        K_pp = torch.stack([
            p_top_k[t] @ self.kernel(p_embed[t], p_embed[t], **kernel_kwargs) @ p_top_k[t].T
            for t in range(i, end_idx)
        ])
        
        K_qq = torch.stack([
            q_top_k[t] @ self.kernel(q_embed[t], q_embed[t], **kernel_kwargs) @ q_top_k[t].T
            for t in range(i, end_idx)
        ])
        
        K_pq = torch.stack([
            p_top_k[t] @ self.kernel(p_embed[t], q_embed[t], **kernel_kwargs) @ q_top_k[t].T
            for t in range(i, end_idx)
        ])
        
        # MMD² = E[k(p,p)] + E[k(q,q)] - 2E[k(p,q)]
        mmd_batch = K_pp + K_qq - 2 * K_pq
        mmd_scores.append(mmd_batch)
    
    return torch.cat(mmd_scores).cpu()
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
for hid in answer_hid:
    # Apply final layer norm and project to vocabulary
    if hasattr(self.model.model, 'language_model'):
        lens_logits = self.model.lm_head(self.model.model.language_model.norm(hid))
    else:
        lens_logits = self.model.lm_head(self.model.model.norm(hid))
    
    logit_lens_res.append(F.softmax(lens_logits, dim=-1))
```

- For each transformer layer, we take the hidden states and project them into vocabulary space
- This reveals what the model "thinks" the next token should be at each layer
- We store the probability distribution over tokens at each layer

Then we compute the **Information Processing Rate (IPR)**:

```python
@torch.no_grad()
def __compute_ipr(self, hid_prob, ans_prob, ans_ids):
    """
    Compute Information Processing Rate (IPR) efficiently using vectorized operations.
    """
    T = ans_prob.shape[0]
    num_layers = len(hid_prob)
    
    # Stack all layer probabilities: (num_layers, T, vocab_size)
    hid_prob_stacked = torch.stack(hid_prob)
    
    # Get max predictions for each token position: (T,)
    max_ids = torch.argmax(ans_prob, dim=-1)
    
    # Compute entropy for all layers and tokens at once: (num_layers, T)
    entropy = self.__compute_entropy(hid_prob_stacked)
    
    # Compute weights (inverse entropy): (num_layers, T)
    weights = 1.0 / (entropy + 1e-8)
    
    # Layer indices (1-based): (num_layers, 1)
    layer_indices = torch.arange(1, num_layers + 1, device=self.device).unsqueeze(1)
    
    # Extract probabilities for max_ids across all layers: (num_layers, T)
    batch_indices = torch.arange(T, device=self.device).unsqueeze(0).expand(num_layers, -1)
    hid_max_probs = hid_prob_stacked[
        torch.arange(num_layers, device=self.device).unsqueeze(1),
        batch_indices,
        max_ids.unsqueeze(0).expand(num_layers, -1)
    ]
    ans_max_probs = ans_prob[batch_indices[0], max_ids]  # (T,)
    
    # Compute ratios: (num_layers, T)
    ratios = 1 - torch.clamp(hid_max_probs / ans_max_probs.unsqueeze(0), max=1.0)
    
    # Weighted layer ratios: (num_layers, T)
    weighted_ratios = ratios * layer_indices
    
    # Sum over layers and normalize: (T,)
    total_weighted_ratio = weighted_ratios.sum(dim=0)
    total_weight = (layer_indices * weights).sum(dim=0)
    
    # Extract answer probabilities: (T,)
    ans_token_probs = ans_prob[batch_indices[0], ans_ids]
    
    # Final IPR computation: (T,)
    ipr = (total_weighted_ratio / total_weight) * (ans_token_probs / ans_max_probs)
    
    return ipr.cpu()
```

- For each token in the generated answer, we compare predictions at each intermediate layer to the final output layer
- If the model's prediction doesn't converge until later layers, it suggests the model is adding more information during processing (likely from internal knowledge)
- We weight deeper layers more heavily (multiplied by `layer_indices`)
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