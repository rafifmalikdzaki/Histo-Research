# §II.A Data Preparation (replacement text for JOIV_revised_v1.docx)

## A. Data Preparation

We use two histopathology sources. The primary cohort (IMI/Hepar) comprises
liver samples from a NAFLD / statin-intervention study under seven treatment
conditions (HFD+atorvastatin, ND+atorvastatin, HFD+vehicle, HFD positive
control, HFD+simvastatin, ND control, ND normal), with diet (ND/HFD),
magnification (10×/20×/40×), specimen identity, and tile index recoverable from
filenames (parsed with zero failures by `src/metadata.py`). Tiling yields 21,168
tiles at 128×128 pixels in RGB; the flat `tiles/` directory is the single source
of truth. The secondary source is PanNuke, a multi-organ histology collection
(7,901 patches including liver, kidney, breast, and other tissues) used for
pre-training, after which the model is adapted to the Hepar cohort.

**Specimen-level splitting.** Tiles from one specimen are near-duplicates, so a
per-tile random split leaks information across folds. We split by specimen with
`StratifiedGroupKFold`, stratified by treatment group, and assert in code that
no specimen appears in both partitions. The held-out test set is 4,536 tiles /
42 specimens; training uses 16,632 tiles / 154 specimens; all seven treatment
groups are present in both. Magnification is stratified across splits and
additionally analysed as an explicit confound (see §III-D).

**Preprocessing.** Tiles are read as RGB and scaled to [0, 1]; no stain
normalisation beyond per-channel scaling is applied. Runs reported here use no
geometric or colour augmentation (the dataset class applies no transform); we
state this plainly and treat augmentation as future work rather than claiming a
robustness benefit we did not train for.

> Honesty note (do not publish, for the response letter): the submitted draft
> claimed rotations/flips/colour-normalisation augmentation and a 70/15/15
> per-tile split. Neither describes the actual runs. Both are corrected above
> (A1); the augmentation sentence is removed, not reworded.
