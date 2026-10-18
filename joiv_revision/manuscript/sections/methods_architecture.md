# §II.B Architecture (replacement text for JOIV_revised_v1.docx)

## B. Proposed Attention-Enhanced Dual Autoencoder

**How the dual autoencoder differs from a conventional autoencoder.** A
conventional autoencoder makes one encode–decode pass
(x → z → x̂) and minimises reconstruction error. Our framework chains two
autoencoder stages: `Autoencoder_Encoder → Autoencoder_Bottleneck →
Autoencoder_Decoder`, assembled as `DAE_KAN_Attention`. The first stage encodes a 128×128×3 patch through three strided convolutions
(3→384→128→64 channels, each followed by batch normalisation and ELU) to a
16×16×64 map, where a KAN convolutional layer (3×3 kernel, B-spline grid 3,
spline order 2) replaces fixed linear kernel weights with learnable univariate
spline functions on each kernel element, so local intensity transitions are
modelled nonlinearly; transposed convolutions plus skip residuals from both
encoder stages then restore a 64×64×384 encoding. The bottleneck compresses
this (384→384→16 channels with BAM attention at both widths) to the latent
representation `z` (16×16×16), and the final decoder reconstructs the patch —
the second pass of the dual structure.

**Attention.** Two complementary mechanisms modulate the features. The
Efficient Channel Attention (ECA) module applies a lightweight 1-D convolution
(kernel 3; γ = 2, b = 1) over channel-pooled descriptors to reweight
informative channels. The Bottleneck Attention Module (BAM) is placed at the
bottleneck and refines both channel and spatial relationships (reduction ratio
16, kernel 3, dilation 4), preserving localised histological cues that plain
pooling would discard.

**Latent spaces.** The DEC training objective operates on a projection head
that maps the flattened bottleneck `z` (16×16×16 = 4096) to a unit-norm vector
of dimension `latent_dim` (default 128), where learnable centroids live. For
evaluation, the dumped test embeddings are the 4096-dimensional bottleneck
`z` itself (standardised + L2-normalised) for every arm, so recon-only and
DEC-trained models are compared in one identical evaluation space; the
projection head shapes training, not the evaluated representation.

**Training.** All variants share one objective:
L = w_rec · MSE(x, x̂) + w_clu · KL(P‖Q), where Q is the Student-t soft
assignment of latent vectors to learnable centroids and P its sharpened target
(DEC/IDEC form). With w_clu = 0 training is pure reconstruction (our baseline
signal); with w_clu > 0 the clustering term is enabled after a reconstruction
warm-up, with centroids KMeans-initialised on warmed-up latents.
Optimisation is AdamW (lr 2e-3, weight decay 1e-5), ReduceLROnPlateau
(factor 0.7, patience 2), batch size 8, global seeding (42/123/456…), early
stopping disabled in favour of a fixed epoch budget with dumped epoch
embeddings for convergence analysis. ELU activation (Eq. 1) is used throughout:

ELU(x) = x if x > 0 else α(exp(x) − 1), with α = 1,

which permits negative outputs and avoids dead units. Seven variants isolate
every component: full `dae_kan_attention`, `baseline` (plain autoencoder),
`bam_only`, `kan_only`, `no_bam`, `no_kan`, `no_eka`, each reporting parameter
counts so module effects are separable from capacity.
