import torch
from torch.nn import functional as F


class LUMINA():
    def __init__(self, model, tokenizer, kernel='cosine', lam=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.lam = lam
        if kernel == "cosine":
            self.kernel = self.__cosine_kernel
        elif kernel == "rbf":
            self.kernel = self.__rbf_kernel
        else:
            raise ValueError("Invalid kernel.")
        
    def __compute_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
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

    def __get_topk_embeddings_and_probs(self, probs, k, embedding_layer):
        T = probs.shape[0]
        top_k_probs = []
        sampled_embeddings = []
        with torch.no_grad():
            for t in range(T):
                topk_probs, topk_ids = torch.topk(probs[t], k, dim=-1)  # (T, k)
                embeddings = embedding_layer(topk_ids)  # (T, k, d)
                top_k_probs.append(topk_probs.to("cuda").float())
                sampled_embeddings.append(embeddings)
        return top_k_probs, sampled_embeddings  # shape: (T, k), (T, k, d)
    
    def __rbf_kernel(self, x, y, sigma=1.0):
        x = x.unsqueeze(1)  # (N, 1, d)
        y = y.unsqueeze(0)  # (1, M, d)
        dist_sq = ((x - y)**2).sum(-1)  # (N, M)
        return torch.exp(-dist_sq / (2 * sigma ** 2)).float()

    def __cosine_kernel(self, x, y, eps=1e-8):
        """
        Compute cosine similarity kernel between two sets of vectors.

        Args:
            x: Tensor of shape (N, d)
            y: Tensor of shape (M, d)
            eps: Small value to avoid division by zero

        Returns:
            A (N, M) tensor where each entry [i, j] = cosine_similarity(x[i], y[j])
        """
        
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)  # (N, d)
        y_norm = y / (y.norm(dim=-1, keepdim=True) + eps)  # (M, d)

        return (1 + torch.matmul(x_norm, y_norm.T).float()) / 2 # (N, M)
    
    def __compute_mmd(self, p_prob, q_prob, embedding_layer, k=100, **kernel_kwargs):
        p_top_k_list, p_embed_list = self.__get_topk_embeddings_and_probs(p_prob, k, embedding_layer)
        q_top_k_list, q_embed_list = self.__get_topk_embeddings_and_probs(q_prob, k, embedding_layer)
        with torch.no_grad():
            K_pp = torch.stack([p_top_k @ self.kernel(p_embed, p_embed, **kernel_kwargs) @ p_top_k.T for p_top_k, p_embed in zip(p_top_k_list, p_embed_list)])
            K_qq = torch.stack([q_top_k @ self.kernel(q_embed, q_embed, **kernel_kwargs) @ q_top_k.T for q_top_k, q_embed in zip(q_top_k_list, q_embed_list)])
            K_pq = torch.stack([p_top_k @ self.kernel(p_embed, q_embed, **kernel_kwargs) @ q_top_k.T for p_top_k, p_embed, q_top_k, q_embed in zip(p_top_k_list, p_embed_list, q_top_k_list, q_embed_list)])
        return (K_pp + K_qq - 2 * K_pq).cpu()
    
    def __get_ans(self, logits, input_ids, prefix_ids, hidden_states=None):
        """
        Get the conditional probability distribution for each token in the response part.
        """
        # Focus only on the RAG-generated response portion
        start = prefix_ids.shape[-1]
        probs = F.softmax(logits[:, start-1:-1, :], dim=-1)  # shift left to align with next-token prediction
        targets = input_ids[:, start:]
        res = (probs.squeeze(0).float(), targets.squeeze(0))
        if hidden_states is not None:
            res += ([hidden_state[:, start-1:-1, :].squeeze(0) for hidden_state in hidden_states], )
        return res

    def __build_input(self, prompt, response):
        messages = [
                    {"role": "user", "content": prompt[:12000]}
                ]
        prefix = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_text = prefix + ' ' + response
        input_ids = self.tokenizer([input_text], return_tensors="pt").input_ids.to("cuda")
        prefix_ids = self.tokenizer([prefix], return_tensors="pt").input_ids.to("cuda")
        return input_ids, prefix_ids

    def predict(self, prompt_w_context, prompt_w_random_context, response):
        input_w_context_ids, prefix_w_context_ids = self.__build_input(prompt_w_context, response)
        input_w_wrong_context_ids, prefix_w_wrong_context_ids = self.__build_input(prompt_w_random_context, response)

        with torch.no_grad():
            outputs_w_context = self.model(
                    input_ids=input_w_context_ids,
                    return_dict=True,
                    output_hidden_states=True,
            )
            
            outputs_w_wrong_context = self.model(
                    input_ids=input_w_wrong_context_ids,
                    return_dict=True,
                    output_hidden_states=True,
            )

        logits_w_context = outputs_w_context['logits']
        hidden_states_w_context = outputs_w_context['hidden_states'][1:]
        logits_w_wrong_context = outputs_w_wrong_context['logits']

        embedding_layer = self.model.get_input_embeddings()  # or any token embedding function
        prob_w_context, answer_ids_w_context, answer_hid_w_context = self.__get_ans(logits_w_context, input_w_context_ids, prefix_w_context_ids, hidden_states_w_context)
        prob_w_wrong_context, _ = self.__get_ans(logits_w_wrong_context, input_w_wrong_context_ids, prefix_w_wrong_context_ids)
        
        mmd = self.__compute_mmd(prob_w_context, prob_w_wrong_context, embedding_layer, k=100)

        logit_lens_res = []
        with torch.no_grad():
            for l, hid in enumerate(answer_hid_w_context):
                if hasattr(self.model.model, 'language_model'):
                    lens_logits = self.model.lm_head(self.model.model.language_model.norm(hid)).float()
                else:
                    lens_logits = self.model.lm_head(self.model.model.norm(hid)).float()
                logit_lens_res.append(F.softmax(lens_logits, dim=-1))
        
        ipr = self.__compute_ipr(logit_lens_res, prob_w_context, answer_ids_w_context)

        return self.lam * ipr - (1 - self.lam) * mmd, mmd, ipr