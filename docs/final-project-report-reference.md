# Investigating Natural Disasters in Nepal

> Reference supplied by Doanh Phung on October 5, 2026. Wording, claims, references, and template instructions are preserved; spacing and tables are formatted for Markdown. This is the supplied report, not evidence that the planned experiments have been completed. See [completion audit](project-completion-audit.md) for repository status and discrepancies.

OpenToWork

Julian Espinal, Evan Demas, Daksh Gajaria, Jessica Keene, Doanh Phung

Group Leader: Julian Espinal

## Problem and Motivation

On 26 August 2026, a glacier collapse near Langtang Lirung sent a debris flood down the Bhote Koshi-Trishuli river corridor. The flood destroyed the Rasuwagadhi-Gyirong border crossing and damaged settlements, roads, bridges, and hydropower facilities across Rasuwa and Nuwakot districts [1, 2]. More than 1,000 people have been confirmed dead and thousands remain missing, making it one of Nepal's worst disasters in years [1]. The event also exposed a monitoring gap. River gauges designed for monsoon floods could not detect the collapse in time, and upstream stations were swept away before they could send alerts [3].

This project discusses whether a convolutional neural network (CNN) land-cover classifier trained on public Sentinel-2 imagery can support that mapping. We classify 64×64-pixel patches at 10 m resolution into land-cover classes such as River, Forest, Residential, and Annual Crop. We then compare predictions on pre- and post-flood scenes to flag where land cover changed. At 10 m resolution, the goal is not to locate individual people or structures. The goal is a fast first-pass map showing where settlements, fields, and vegetation were replaced by water or debris.

The problem is nontrivial for three reasons. First, labeled satellite data for Nepal is scarce, so the model must be trained elsewhere. EuroSAT, the standard Sentinel-2 benchmark, covers only European land cover [4]. Secondly, Himalayan terrain looks unlike anything in that training set: terraced hillsides, landslide scars, braided riverbeds, and glaciers. High accuracy on a European test split may therefore overstate real-world performance. Third, Sentinel-2's infrared bands carry information about water and vegetation that RGB images lack. It is unclear how much a CNN gains from them, especially when training data is limited.

Deep learning is appropriate here for two reasons. CNNs learn spatial and spectral features directly from pixels and have reached roughly 98% accuracy on EuroSAT, well above classical feature-based baselines [4]. Transfer learning from pre-trained networks is also a standard remedy for limited labeled data.

## Research questions

RQ1 (data efficiency). How does test macro-F1 change as the training set shrinks from 100% to 10%, 5%, and 1%? Does a pre-trained CNN degrade less than the same architecture trained from scratch? Hypothesis: pre-training's advantage widens as training data decreases.

RQ2 (spectral input). Do all 13 Sentinel-2 bands improve macro-F1 over RGB alone, and are the gains concentrated in particular classes? Hypothesis: multispectral input helps most on water and vegetation classes (River, Sea/Lake, Forest, Herbaceous Vegetation).

RQ3 (generalization to Nepal). How much does performance drop from the EuroSAT test set to hand-labeled pre-event patches from the Trishuli corridor? Do pre/post-event prediction changes align with published impact maps? Hypothesis: performance drops substantially, with errors concentrated in classes whose Himalayan appearance differs most from Europe.

## Proposed solution and course connection

Task and model family. We treat each 64×64 Sentinel-2 patch as an image classification problem with 10 output classes (the EuroSAT land-cover classes). Our primary model is ResNet-18 [5], a residual CNN with about 11M parameters. We train it in two ways: fine-tuned from ImageNet-pretrained weights, and trained from random initialization with an identical architecture. Keeping the architecture fixed means any performance difference between the two comes from pre-training alone, which connects back to RQ1. We also train a compact custom CNN (under 1M parameters) as a lightweight baseline. It has four blocks of [3×3 convolution → batch normalization → ReLU → 2×2 max pooling] with 32, 64, 128, and 256 filters, followed by global average pooling, dropout (p = 0.5), and a fully connected output layer. This model is used for the regularization ablation and as our fallback.

Inputs and outputs. The input is a C×64×64 tensor, where C = 3 for RGB (bands B4, B3, B2) or C = 13 for all Sentinel-2 bands. Each band is standardized using training-set statistics only. Patches are kept at 64×64, so ResNet-18's final feature map is 2×2 before global average pooling. The output is a softmax distribution over the 10 classes.

Loss and optimization. We use cross-entropy loss. Class imbalance in EuroSAT is mild, so we do not weight the classes. We optimize with mini-batch AdamW (batch size 64, weight decay 1e-4) and a cosine learning-rate schedule with a short warm-up. Learning rates are:

- 1e-3 for models trained from scratch
- 1e-4 for the pretrained backbone and 1e-3 for the new classifier head during fine-tuning

Each training-set fraction gets the same number of optimizer steps rather than the same number of epochs. Otherwise the 1% subset (about 160 images) would receive too few updates to compare fairly. For every run, we keep the checkpoint with the best validation macro-F1.

Regularization and design choices.

Data augmentation (training data only). We use random horizontal and vertical flips and 90° rotations, since overhead imagery has no natural "up." We do not use color jitter, because it would distort the spectral signatures the multispectral model relies on.

Other regularizers. We use batch normalization, dropout before the classifier, weight decay, and early stopping on the validation set.

Regularization ablation. On the compact CNN at the 10% data fraction, we switch augmentation and dropout on and off. This measures each technique's effect on validation macro-F1 and on the train/validation generalization gap.

Repeated runs. Every configuration is run with 3 random seeds, so we can report a mean and standard deviation.

Experiment grid. The main grid for RQ1 and RQ2 is 2 initializations × 2 input types × 4 data fractions × 3 seeds, which comes to 48 ResNet-18 runs. Because the patches are small (64×64), each run should take minutes on a single GPU.

Applying the model to Nepal (RQ3). We choose the best configuration using validation results, then apply it without further training to the hand-labeled pre-event patches from the Trishuli corridor. This gives a measure of accuracy under domain shift. Next, we run the same model on co-registered pre- and post-event tiles. We flag a patch as "changed" when its predicted class differs between the two dates. We also flag patches where the model's maximum softmax probability is low, since those are likely land-cover types the model never saw in training. The flagged patches form a change map that we compare against published impact maps.

## Data Plan

Primary dataset: EuroSAT. It contains 27,000 labeled Sentinel-2 patches (64×64 pixels, 10 m resolution) across 10 land-use classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, Pasture, Permanent Crop, Residential, River, and Sea/Lake. The patches were collected from cities across Europe. EuroSAT is available in an RGB version and a 13-band multispectral version. It is freely downloadable from the authors' GitHub repository, through torchvision.datasets.EuroSAT (RGB), and through TorchGeo (multispectral). It is released under the MIT license.

Class balance. Each class has between about 2,000 and 3,000 images, so imbalance is mild. We will still report macro-F1 and per-class results so that smaller classes are not hidden.

Splits. EuroSAT has no single official split. We will use a fixed, stratified 60/20/20 train/validation/test split and save the indices to the repository (or TorchGeo's standard split if we confirm it is stratified). For RQ1, we will draw stratified subsets of 1%, 5%, 10%, and 100% of the training split, using the same subsets for every model.

Preprocessing. Per-band standardization (training statistics only), matching band order between the RGB and multispectral versions, and augmentation applied to training data only. Leakage risk: EuroSAT patches can come from neighboring locations, which makes random splits somewhat optimistic. We will note this as a limitation, and it is part of why the Nepal test in RQ3 matters.

Nepal case-study data. We will export Sentinel-2 imagery of the Trishuli/Bhote Koshi corridor (Rasuwa and Nuwakot districts) from Google Earth Engine, choosing one cloud-free pre-event scene and one post-event scene. We will use Level-1C top-of-atmosphere data to match how EuroSAT was built, with the same 13 bands, and tile the scenes into 64×64 patches. Two team members will hand-label about 300–500 pre-event patches with the closest EuroSAT class, excluding patches that fit no class (for example glaciers and bare rock). We will compare pre/post predictions against published impact maps from sources such as USGS, UNOSAT, or Copernicus EMS, if available.

Data risks. (1) Cloud cover: the flood happened during the monsoon, so cloud-free post-event optical imagery may be scarce. (2) Label mismatch: Himalayan land cover such as terraces, landslide scars, and glaciers does not map cleanly onto EuroSAT's European classes, and our hand labels will be imperfect. (3) All imagery is public satellite data at 10 m resolution, so there are no personal-data concerns.

## Model Evaluation and analysis

The main criterion we will use for our evaluation is macro-F1, as this measure gives equal weight to all ten EuroSAT classes and ensures that differences due to class frequency do not have a dominant effect on the results. We will also provide the overall accuracy as well as the precision, recall, and F1 score for each individual class, together with the confusion matrices. For each of the main configurations, training will be carried out using three different random seeds, and we will give the mean performance together with the standard deviation.

With respect to RQ1, we plan to compare the pretrained and randomly initialized ResNet-18 models when the training fractions are 100%, 10%, 5%, and 1%. Evidence in favor of our hypothesis would be a growing macro-F1 advantage for the pretrained model as the amount of training data shrinks. To tell whether the issues encountered are due to optimization problems or to overfitting, we will examine the learning curves and the gap between training and validation performance.

In order to answer RQ2, we will make a comparison between the RGB and the 13-band models using macro-F1 and per-class F1. We will focus on the River, Sea/Lake, Forest, and Herbaceous Vegetation classes to find out if multispectral information leads to greater improvements for the water and vegetation classes.

For RQ3, we will evaluate the selected model on the independently hand-labeled Nepal patches and compare its macro-F1 and per-class performance with its EuroSAT test performance. The difference between these results will quantify the effect of domain shift. We will also inspect confusion matrices and representative failure cases and compare predicted pre/post-event changes with available published impact maps.

Finally, the compact CNN regularization ablation will compare augmentation and dropout configurations using validation macro-F1 and train-validation generalization gap. Expected outputs include learning curves, macro-F1 comparison tables, confusion matrices, per-class F1 plots, and the Nepal change map.

## Feasibility, resources, and risks

The use of computing and software means that the project can be carried out with a single GPU, for example a GPU available on Google Colab or another GPU that is compatible with CUDA. Since EuroSAT has only 27,000 images of size 64×64 and our proposed experiment involves 48 runs of ResNet-18, and given that the images are small and ResNet-18 is relatively light in terms of requirements, each individual training run is expected to take only a few minutes rather than hours on a GPU. We will mainly use Python and PyTorch, together with Torchvision for the RGB EuroSAT dataset and the pre-trained ResNet-18 model, TorchGeo for the 13-band multispectral data, and standard libraries such as NumPy, pandas, Matplotlib and scikit-learn for preprocessing and evaluation.

The main external dependencies include access to the EuroSAT RGB and multispectral datasets as well as the use of Google Earth Engine to obtain Sentinel-2 imagery of the study area in Nepal. For the part relating to Nepal, it is necessary to locate appropriate pre- and post-event images and, where available, to use published impact maps from organisations such as USGS, UNOSAT, or Copernicus EMS for comparison.

The main risks together with a backup plan. The greatest data risk is cloud coverage over Nepal since the flood took place during the monsoon season, which could make it difficult to get a usable cloud-free Sentinel-2 image after the event. The second major risk is domain shift and a mismatch of labels: EuroSAT includes imagery from Europe, whereas Nepal has Himalayan features such as terraced hillsides, landslide scars, glaciers, and bare rock, none of which may suit the dataset's 10 classes. If it is not possible to obtain suitable imagery or labels for Nepal, then the Nepal change-detection analysis will be treated as optional and the project will proceed using the EuroSAT experiments. We will still be able to carry out a full investigation into data efficiency, transfer learning, the use of RGB versus 13-band multispectral input, and the application of regularization with ResNet-18 and the compact CNN. In this way, a complete deep-learning project comprising quantitative evaluation, ablation studies, learning curves, per-class metrics, and a failure-case analysis is maintained even if the Nepal case study cannot be carried out.

## Team Responsibilities

List every member and their initial technical responsibilities. Responsibilities may overlap, but each student must own substantive technical work; do not assign one member only writing/presentation duties.

| Team member | Initial Technical Responsibilities |
| --- | --- |
| Julian Espinal | Build the EuroSAT RGB download and preprocessing pipeline; create and save the stratified splits and training subsets; audit class balance and data leakage risks. |
| Evan Demas | Implement ResNet-18 training from scratch and ImageNet fine-tuning; run the data-efficiency experiments for RQ1; save checkpoints and learning curves. |
| Doanh Phung | Design the reproducible experiment and metric pipeline; quantify performance, uncertainty, and domain shift across all models; evaluate change-detection maps |
| Daskh Gajaria | Build the 13-band TorchGeo pipeline; adapt ResNet-18 for multispectral input; run the RGB-versus-multispectral experiments for RQ2. |
| Jessica Keene | Acquire and co-register the pre/post Nepal imagery through Google Earth Engine; develop the patch-labeling protocol; coordinate domain-shift labels and reference impact maps. |
| All team member | Review Nepal labels, reproduce at least one experiment, interpret failure cases, contribute to the final report, and prepare for the presentation. |

## Project Timeline

Provide a date-based or week-by-week plan through project completion. For each milestone, give an owner, expected output, and dependency. Reserve time for debugging, evaluation, error analysis, and writing.

| Dates | Milestone and owner(s) | Expected output | Dependency |
| --- | --- | --- | --- |
| Sept. 22 | Finalize proposal and repository - All | Submitted proposal, shared repository, experiment plan, and assigned responsibilities | Team agreement |
| Sept. 23–24 | Data and preprocessing - Julian, Daksh | Downloaded RGB/13-band EuroSAT data, saved splits/subsets, normalization statistics, and tested data loaders | Dataset access |
| Sept. 23–25 | Model setup -Evan, Daksh, Doanh | Working ResNet-18, compact CNN, metric code, configuration files, and checkpoint logging | Functional data loaders |
| Sept. 24–28 | Nepal data preparation -Jessica, Julian | Co-registered pre/post scenes, tiled patches, labeling guide, and initial labeled evaluation set | Earth Engine imagery availability |
| Sept. 26–27 | First complete training run -Evan, Doanh | End-to-end baseline run with learning curves, validation metrics, and saved checkpoint | Model and data pipeline |
| Sept. 28–Oct. 1 | Main experiments - Evan, Daksh, Doanh | RQ1/RQ2 experiment grid across data fractions, input types, initializations, and seeds | Validated baseline |
| Oct. 1–3 | Ablations and Nepal evaluation - Jessica, Doanh, Julian | Regularization ablation, domain-shift metrics, change map, and preliminary failure analysis | Main runs and Nepal labels |
| Oct. 4–5 | Final model selection and test evaluation - All | Frozen model choice, one final test evaluation, result tables, and confusion matrices | Completed validation experiments |
| Oct. 6–7 | Integration, debugging, and writing - All | Reproducibility check, final figures, error analysis, report, and presentation draft | Final metrics |
| Oct. 8 | Final submission - All | Complete project materials and reproducible repository | Final review |
| Oct. 9 | Presentation - All | Presentation of problem, method, experiments, results, limitations, and conclusions | Submitted project |

## References

Minimum 3 relevant sources: the dataset source, sources used to motivate the problem or model choice, and any important code/pre-trained model source used in the plan. Use one consistent citation style

[1] A. Abidi, T. Siu, and S. Bajracharya, “In a Nepal town, flood survivors find little trace of former lives,” Reuters, Sept. 18, 2026. [Online]. Available: https://www.reuters.com/business/environment/nepal-town-flood-survivors-find-little-trace-former-lives-2026-09-18/ [Accessed: Sept. 22, 2026].

[2] S. Arasu, “Climate change made deadly Nepal glacier and rock collapse more likely, according to analysis,” Associated Press, Sept. 17, 2026. [Online]. Available: https://apnews.com/article/nepal-floods-climate-change-65ee7e50dfe7914dff37d893a5e6b810 [Accessed: Sept. 22, 2026].

[3] S. Chaganti Singh, “Nepal to rebuild early-warning system on China border after deadly flood, sources say,” Reuters, Sept. 2, 2026. [Online]. Available: https://www.reuters.com/business/environment/nepal-rebuild-early-warning-system-china-border-after-deadly-flood-sources-say-2026-09-02/ [Accessed: Sept. 22, 2026].

[4] P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217–2226, 2019, doi: 10.1109/JSTARS.2019.2918242.

[5] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016, doi: 10.1109/CVPR.2016.90.

[6] N. Gorelick, M. Hancher, M. Dixon, S. Ilyushchenko, D. Thau, and R. Moore, “Google Earth Engine: Planetary-scale geospatial analysis for everyone,” Remote Sensing of Environment, vol. 202, pp. 18–27, 2017, doi: 10.1016/j.rse.2017.06.031.

[7] A. J. Stewart, C. Robinson, I. A. Corley, A. Ortiz, J. M. Lavista Ferres, and A. Banerjee, “TorchGeo: Deep Learning With Geospatial Data,” in Proceedings of the 30th International Conference on Advances in Geographic Information Systems, Article 19, pp. 1–12, 2022, doi: 10.1145/3557915.3560953.

[8] PyTorch, “resnet18 — Torchvision documentation.” [Online]. Available: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html [Accessed: Sept. 22, 2026].

[9] P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,” dataset, version 2, Zenodo, 2018, doi: 10.5281/zenodo.7711810.
