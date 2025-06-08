## Application Focused Machine Learning

The purpose of this repository is to bridge the gap between ML as taught in university courses, and its practical applications in the workforce, research, or broadscale problem solving. Rather than separating techniques by architectural similarity, I outline the broad problem domains (“use cases”) in machine learning and, for each, cover the main families of models or methods that are commonly applied. Each use case serves as an entry point—when you know *what* you want to achieve, you can quickly see which techniques are most appropriate. I feature newer techniques in in ML Cloud Systems and Generative AI not covered in most classrooms.

*Note: for a comprehensive review on Adversarial AI, please see my other repo, [https://github.com/Ananya-Krishna/Adversarial_AI_Attacks](https://github.com/Ananya-Krishna/Adversarial_AI_Attacks)*
_______

# Table of Contents:

### 1. Supervised Learning (Regression & Classification)

- **Tabular Regression**  
  - *Ridge / Lasso / Elastic Net* — linear models with ℓ₂/ℓ₁ regularization for high-dimensional or collinear features.  
  - *Random Forest / Gradient Boosting (XGBoost, LightGBM)* — tree-based ensembles that handle nonlinearities and interactions out of the box.  
  - *Gaussian Processes* — nonparametric Bayesian models providing uncertainty estimates over continuous outputs.

- **Tabular Classification**  
  - *Logistic Regression* — simple baseline for binary or multiclass with probabilistic outputs.  
  - *Support Vector Machines (SVM)* — can incorporate kernel functions to capture complex decision boundaries.  
  - *k-Nearest Neighbors (KNN)* — distance-based classification, often used as a quick baseline.  
  - *Tree-Based Methods* (Decision Trees, Random Forest, Gradient Boosting) — handle heterogeneous features and missing values effectively.

- **Image Classification**  
  - *Convolutional Neural Networks (CNNs)* (ResNet, EfficientNet, etc.) — exploit spatial locality to learn hierarchical features.  
  - *Vision Transformers (ViT)* — apply self-attention to image patches, often pretrained on large datasets.  
  - *Transfer Learning* — fine-tune pretrained backbones (e.g., ResNet, Inception) on downstream tasks with limited data.

- **Text Classification**  
  - *Bag-of-Words + Linear Models* — TF-IDF or word embeddings fed into logistic regression or SVM.  
  - *RNN / LSTM / GRU* — sequence models that capture order information, useful for moderate-length texts.  
  - *Transformer-Based Models* (BERT, RoBERTa) — leverage pretrained contextual embeddings for state-of-the-art performance.

---

### 2. Generative Modeling

- **Text Generation (Autoregressive)**  
  - *GPT-Style Transformers (Causal Masking)* — predict the next token in a sequence, pretrained on massive corpora.  
  - *RNN / LSTM-Based Language Models* — older sequence models that can still be effective on smaller datasets.
  - *Diffusion Models for Text* — use iterative denoising to generate coherent text (emerging research area).

- **Diffusion & Energy-Based Models**  
  - *Denoising Diffusion Probabilistic Models (DDPM)* — learn to reverse a noise process to generate new samples.  
  - *Latent Diffusion* — perform diffusion in a lower-dimensional latent space for efficiency.

- **Variational Autoencoders (VAEs) & Normalizing Flows**  
  - *VAEs (β-VAE, Conditional VAE)* — combine reconstruction loss with KL divergence to learn structured latent representations.  
  - *Normalizing Flows (RealNVP, Masked Autoregressive Flows)* — learn invertible mappings for exact likelihoods and latent interpolation.

- **GAN Variants (Optional)**  
  - *Basic GAN, WGAN, StyleGAN, etc.* — adversarial framework for mapping random noise to realistic samples (images, audio, etc.).

---

### 3. Representation Learning & Embeddings

- **Self-Supervised Vision**  
  - *SimCLR, MoCo, BYOL* — contrastive methods that learn visual features without labels by maximizing agreement between augmented views.

- **Contrastive Language–Vision**  
  - *CLIP, ALIGN* — align image and text embeddings via a contrastive objective, enabling zero-shot classification and retrieval.

- **Multimodal Retrieval & Generation**  
  - *Retrieval-Augmented Generation (RAG)* — combine pretrained language models with external retrieval components to ground generation.  

- **Graph Neural Networks (GNNs)**  
  - *Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), Hyperbolic GNNs* — learn node/graph embeddings for structured data such as social networks, biological networks, or knowledge graphs.

---

### 4. Reinforcement Learning & Decision Making

- **Value-Based Methods**  
  - *Q-Learning, Deep Q-Networks (DQN)* — learn action-value functions to choose actions by maximizing expected future reward.  
  - *Double DQN, Dueling DQN* — address overestimation and improve stability.

- **Policy-Based & Actor-Critic Methods**  
  - *REINFORCE, A2C / A3C* — optimize the policy directly using gradient estimates of expected reward.  
  - *Proximal Policy Optimization (PPO), Trust Region Policy Optimization (TRPO)* — improve stability via clipped or constrained policy updates.

- **Advanced & Continuous Control**  
  - *Soft Actor-Critic (SAC), Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3)* — handle continuous action spaces with off-policy actor-critic frameworks.  
  - *Generalized Policy Optimization (GRPO)*, *Inverse RL*, *Linearly Solvable MDPs (LMDPs)* — specialized methods for specific tasks or theoretical guarantees.

---

### 5. Sequence Models & Time Series

- **Markov-Based Models**  
  - *Hidden Markov Models (HMMs)* — probabilistic sequence models with discrete latent states.  
  - *Kalman Filters* (and variants) — estimate latent state for linear dynamical systems with Gaussian noise.

- **RNN / LSTM / GRU Models**  
  - *Recurrent Neural Networks (RNNs)* — basic sequence learners; susceptible to vanishing gradients on long sequences.  
  - *Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)* — gated architectures that mitigate vanishing/exploding gradient issues, used in language modeling, speech recognition, and time-series forecasting.

- **Transformer Variants for Sequences**  
  - *Standard Transformers (BERT, T5)* — bidirectional attention for masked language modeling or sequence-to-sequence tasks.  
  - *Autoregressive / Causal Transformers (GPT)* — predict next token based on past tokens.  
  - *Long-Context Models (Transformer-XL, Reformer)* — handle very long sequences more efficiently through memory caches or sparse attention.

---

### 6. Computer Vision Beyond Classification

- **Object Detection & Segmentation**  
  - *Faster R-CNN, YOLO, SSD* — detect and localize objects in images in real time or near real time.  
  - *U-Net, Mask R-CNN, DeepLab* — semantic and instance segmentation architectures that delineate object boundaries or pixel-level classes.

- **3D Reconstruction & Rendering**  
  - *Neural Radiance Fields (NeRF)* — learn continuous volumetric scene representations to render novel views.  
  - *Gaussian Splatting* — represent 3D scenes as collections of Gaussian points for fast rendering and editing.  

---

### 7. Explainability & Interpretability (Applied)

- **Vision Explainability**  
  - *Grad-CAM, Guided Backpropagation, Excitation Backpropagation* — generate saliency maps to highlight image regions influencing a CNN’s decision.  
  - *LIME / SHAP for Vision* — approximate local feature importance by perturbing patches or superpixels.

- **Text & Language Explainability**  
  - *Attention Visualization* — inspect attention weights in Transformers to see which tokens influence each output.  
  - *LIME / SHAP for Text* — perturb tokens or n-grams to measure impact on classification or generation decisions.

- **Graph & Structured Data Explainability**  
  - *PG-Explainer, GNNExplainer* — identify subgraphs or node features driving a GNN’s prediction.  
  - *Observable Propagation* — trace activation pathways in large pretrained models to determine importance.

- **Circuit Tracing & Sparse Autoencoders**  
  - *Circuit Tracing* — follow neuron-by-neuron activations in Transformers to understand emergent features.  
  - *Sparse Autoencoders* — learn sparse latent representations that can reveal salient input features.

---

### 8. ML Cloud Systems & Production

- **Model Deployment & Serving**  
  - *TensorFlow Serving, TorchServe, NVIDIA Triton Inference Server* — host and serve models at scale with REST or gRPC endpoints.  
  - *Kubeflow, MLflow, TFX* — orchestration frameworks for end-to-end ML pipelines (data ingestion → training → validation → deployment).

- **Managed Services & AutoML**  
  - *AWS SageMaker, Google AI Platform, Azure ML* — fully managed training, hyperparameter tuning, and deployment services.  
  - *AutoML Frameworks (AutoGluon, AutoKeras, H2O AutoML)* — automated search over model architectures and hyperparameters.

- **Scalable Training & Distributed Systems**  
  - *Horovod, DeepSpeed, FairScale* — frameworks for data-parallel and model-parallel training across multiple GPUs or nodes.  
  - *ZeRO, FSDP* — techniques for memory-efficient large model training via sharded optimizers and parameters.

- **Monitoring & Observability**  
  - *Prometheus / Grafana* — track latency, throughput, and resource usage in production.  
  - *Fairness / Bias Monitoring (e.g., IBM AIF360 integration)* — detect distribution shifts or performance disparities across subgroups.

---

### 9. Loss Functions & Optimizers (Applied Reference)

- **Regression Losses**  
  - *Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss* — trade-offs between sensitivity to outliers and smoothness.  
  - *ε-Insensitive Loss (SVR)* — ignore small deviations, penalize larger errors.

- **Classification Losses**  
  - *Cross-Entropy Loss* — standard for multiclass and binary classification.  
  - *Focal Loss* — down-weights easy examples to focus on hard, minority classes.  
  - *Hinge Loss (SVM)* — margin-based loss for maximum-margin classifiers.

- **Specialized Losses**  
  - *Dice Loss / Jaccard Loss* — measure overlap for segmentation tasks, address class imbalance.  
  - *Evidence Lower Bound (ELBO)* — VAE objective combining reconstruction and KL divergence terms.  
  - *KL Divergence* — measure discrepancy between two distributions (used in Bayesian inference, VAEs).  
  - *Double Descent Discussion* — phenomenon where test error initially decreases, then increases, then decreases again as model capacity grows.

- **Optimizers**  
  - *Stochastic Gradient Descent (SGD)* — base optimizer, often with momentum and learning-rate schedules.  
  - *Adam / AdamW, RMSProp, Adagrad* — adaptive-learning-rate methods that often converge faster in practice.  
  - *LARS, LAMB* — used for scaling up to very large batch sizes (common in large-scale training).  
  - *Learning-Rate Schedules* — OneCycle, Cosine Annealing, Warmup strategies to improve convergence and generalization.

- **First-Order vs. Second-Order Optimization Methods**
  - **First-Order Methods**  
    - *Definition*: Rely exclusively on first derivatives (gradients ∇𝑓(θ)) of the loss function to guide parameter updates.  
    - *Examples*:  
      - **SGD (Stochastic Gradient Descent)**:  
        \[
          θ_{t+1} = θ_{t} - η \,\nabla_θ \mathcal{L}(θ_{t}, x_t)
        \]  
      - **Momentum / Nesterov Accelerated Gradient**:  
        \[
          v_{t+1} = μ\,v_{t} + η \,\nabla_θ \mathcal{L}(θ_{t}), \quad
          θ_{t+1} = θ_{t} - v_{t+1}
        \]  
      - **Adam / AdamW / RMSProp / Adagrad**:  
        \[
          m_{t+1} = β_1\,m_t + (1-β_1)\,\nabla_θ \mathcal{L}(θ_t), 
          \quad
          v_{t+1} = β_2\,v_t + (1-β_2)\,(\nabla_θ \mathcal{L}(θ_t))^2
        \]
        \[
          \hat{m}_{t+1} = \frac{m_{t+1}}{1 - β_1^t}, \quad
          \hat{v}_{t+1} = \frac{v_{t+1}}{1 - β_2^t}, \quad
          θ_{t+1} = θ_t - η \,\frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + ε}
        \]
    - *Advantages*:  
      - Low per-iteration cost (only need gradients, no Hessians).  
      - Scales to large datasets and high-dimensional parameter spaces.  
      - Adaptive methods (Adam, RMSProp) can converge faster in practice without extensive LR tuning.  
    - *Limitations*:  
      - May converge slowly near saddle points or flat regions (“zig-zagging”).  
      - Sensitive to learning‐rate schedules; tuning can be time-consuming.  
      - Cannot leverage curvature information explicitly, so may require more iterations on ill-conditioned problems.

  - **Second-Order Methods**  
    - *Definition*: Use both first derivatives (gradients) and second derivatives (Hessian ∇²𝑓(θ) or approximations) to update parameters, effectively modeling local curvature.  
    - *Examples*:  
      - **Newton’s Method**:  
        \[
          θ_{t+1} = θ_{t} - \bigl[∇^2_θ \,\mathcal{L}(θ_{t})\bigr]^{-1} \,∇_θ \mathcal{L}(θ_{t})
        \]
      - **Quasi-Newton Methods**  
        - *BFGS / L-BFGS*: build an approximate inverse Hessian \(H_t ≈ (∇^2 \mathcal{L}(θ_t))^{-1}\) from gradient differences.  
        - Update rule (for L-BFGS with limited memory):  
          \[
            θ_{t+1} = θ_t - η\,H_t \,∇_θ \mathcal{L}(θ_t)
          \]
      - **Conjugate Gradient (CG)**: solve \(H\,p = -g\) for the search direction \(p\) without fully inverting Hessian, then set
        \[
          θ_{t+1} = θ_t + η\,p
        \]
    - *Advantages*:  
      - Faster convergence in well-conditioned, moderate-scale problems (fewer iterations to reach optimum).  
      - Incorporates curvature, so can “jump” directly to a minimum (quadratic convergence near optimum).  
    - *Limitations*:  
      - High per-iteration cost: storing or inverting a full Hessian is \(O(n^2)\) memory and \(O(n^3)\) compute for \(n\) parameters.  
      - In deep learning, \(n\) is often millions—impractical to compute or store exact Hessian.  
      - Quasi-Newton and CG alleviate some cost but still require storing gradient history or performing multiple Hessian–vector products.  
      - More sensitive to noisy or non‐convex losses; may diverge if curvature estimate is poor.

- **Hybrid “Quasi‐Second‐Order” in Deep Learning**
  - *K-FAC (Kronecker-Factored Approximate Curvature)* — approximates Fisher information matrix in a block-wise fashion for faster natural-gradient updates.  
  - *L-BFGS on Small Fine‐Tuning Tasks* — sometimes used when dataset is small or network head is shallow.

- **Selecting Optimizers in Practice (Applied Guidance)**
  - **When to Use First‐Order**  
    - Training large CNNs or Transformers on millions of samples.  
    - Rapid prototyping or when compute resources are limited.  
    - Tasks where adaptive methods (Adam/AdamW) reliably converge without extensive tuning.
  - **When to Consider (Quasi-)Second‐Order**  
    - Fine‐tuning a small portion of a pretrained model (e.g., last layer) where parameter count is moderate.  
    - Convex subproblems (e.g., logistic regression on top of fixed features).  
    - Research experiments that specifically analyze curvature effects or require high-precision solutions.
---
