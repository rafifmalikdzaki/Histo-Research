# Discussion + Limitations (replacement/adaptation text)

## Discussion

Three corrections change what this study can honestly claim relative to the
submitted draft. First, the leak-free specimen-level split removes the
optimistic bias of per-tile random splitting; every number reported here is on
tiles from specimens the model never saw. Second, external validation is now
genuine: filenames encode treatment group, diet, magnification, and specimen,
so cluster↔biology agreement (AMI/ARI) measures something real instead of
agreement with placeholder labels. Third, the deep-clustering objective gives
the model a reason to be separable: under pure reconstruction the proposed
architecture is not clearly better than a plain autoencoder on internal
indices, and we state this plainly. Its value emerges where the reviewers asked
for it — the latent space agrees more strongly with treatment biology, and the
DEC-style objective markedly improves separability — and the per-magnification
control is what licenses the biological reading, since pooled magnification
association alone would make that claim unsafe.

We therefore frame the contribution as a clustering-objective-trained,
biologically validated phenotyping pipeline rather than an architectural win in
the abstract: architecture drives biological agreement; the training objective
drives separability.

## Limitations

Single-centre cohort (≈196 specimens); magnification correlates with treatment
in the cohort, which is why the stratified analysis is primary, not
supplementary. No geometric/colour augmentation was used in the reported runs.
Pathologist scoring of cluster montages is ⟨available and reported / exported
with the scoring template and pending, in which case this work reports
enrichment statistics and plainly labels expert validation as outstanding⟩.
Self-supervised comparisons are limited to ⟨frozen-encoder + KMeans minimum /
SimCLR/BYOL/VAE where completed⟩ through the identical eval path; a broader
SSL benchmark remains future work alongside multi-centre validation and
stain-normalisation ablations.

## Conclusion (one paragraph)

On a leak-free, confound-controlled benchmark, a deep-clustering objective on
a true latent bottleneck makes an attention/KAN autoencoder produce
well-separated liver-histopathology phenotypes that track treatment and diet
and survive a magnification confound control. ⟨Final multi-seed headline
numbers at freeze.⟩
