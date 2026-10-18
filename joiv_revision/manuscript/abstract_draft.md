# Abstract draft (JOIV: 230–250 words; objectives / materials / method / results / implication)

> Histopathology image analysis is central to understanding liver tissue
> morphology and treatment-induced change, yet manual inspection is
> time-consuming, subject to inter-observer variability, and overwhelmed by the
> heterogeneity of whole-slide data. This study develops an unsupervised
> phenotyping pipeline for liver histopathology from a NAFLD / statin-intervention
> cohort (21,168 tiles across ~196 specimens in seven treatment arms), split by
> specimen with treatment stratification so no specimen appears in both training
> (16,632 tiles / 154 specimens) and test (4,536 tiles / 42 specimens). Our model,
> DAE-KAN-Attn, is a dual convolutional autoencoder that replaces standard
> convolutions with Kolmogorov–Arnold Network layers, adds Bottleneck Attention
> and Efficient Channel Attention at a true unit-norm latent bottleneck, and
> trains with a composite reconstruction plus DEC-style deep-clustering objective.
> Latent embeddings are clustered with KMeans, Bisecting KMeans, and Gaussian
> Mixture Models and validated internally (Silhouette, Davies–Bouldin,
> Calinski–Harabasz, Xie–Beni) and externally against treatment group and diet
> (AMI/ARI/NMI), with magnification held fixed as a confound control.
> ⟨RESULTS: 1–2 sentences with frozen multi-seed numbers, e.g. treatment-AMI and
> separability deltas with CIs.⟩ The treatment signal survives the
> per-magnification control, indicating discovered phenotypes reflect biology
> rather than imaging artifact. This provides a reproducible, statistically
> grounded unsupervised framework for treatment-related morphological discovery
> that can be extended with expert annotation and multi-centre validation.

Word count (excluding ⟨RESULTS⟩ placeholder): 228 — final version must land 230–250
inclusive of the results sentences; trim or extend at freeze.
