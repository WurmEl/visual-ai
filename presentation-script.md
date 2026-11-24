# Presentation Script — Fine-grained Image Aesthetic Assessment

## Slide 1 — Title and Introduction
**0:00–0:30 | Switch to slide 2 at 0:30**

Hello everyone.

Today, I will present the CVPR 2026 paper *Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks*, by Zhichao Yang and colleagues.

The paper investigates how artificial intelligence can evaluate the aesthetic quality of images, especially when the differences between the images are very small.

I will introduce the research problem, explain the proposed model and its three main components, discuss the experimental results, and finally present my proposed follow-up experiment using food photography.

---

## Slide 2 — What Is Image Aesthetic Assessment?
**0:30–1:45 | Switch to slide 3 at 1:45**

First, what exactly is image aesthetic assessment?

In traditional image classification, our goal is to identify what an image contains.

For example, a classifier might recognize a cat, a car, or a mountain.

Image aesthetic assessment, or IAA, asks a different question. Instead of identifying the content, we want to evaluate how visually appealing an image is.

This could depend on several factors, such as composition, lighting, color balance, sharpness, and the overall visual impression.

For example, the two landscape photographs on this slide show almost the same subject, but we might prefer one because of its lighting or composition.

An aesthetic assessment model typically predicts a continuous quality score rather than assigning a category.

Depending on the method, models can also predict a distribution of human ratings or compare the relative quality of two images.

One important point is that aesthetics are subjective. There isn't necessarily one objectively correct answer.

Instead, these models try to learn general patterns from human preferences.

This is already an interesting computer vision problem, but the paper focuses on an even more challenging variation.

---

## Slide 3 — Fine-Grained Aesthetic Differences
**1:45–3:00 | Switch to slide 4 at 3:00**

The authors distinguish between coarse-grained and fine-grained image aesthetic assessment.

In coarse-grained assessment, the differences between images can be quite obvious.

For example, comparing a blurry and badly exposed photograph with a professional landscape picture is relatively straightforward.

Fine-grained assessment is different.

Here, we compare images that show almost exactly the same scene.

As illustrated on this slide, we might have several photographs of the same building, with slightly different framing or perspective.

The differences can also involve small variations in exposure, sharpness, or the position of individual objects.

The challenge is that many existing vision models focus on the overall content of an image.

When visual information is aggregated into a compact representation, small local differences may become less influential.

Another problem is the training data.

If we ask people to rate two nearly identical photographs separately, their numerical ratings might be very close or inconsistent.

However, they may still have a clear preference when comparing the two directly.

So the central question becomes:

How can we train a model to recognize these subtle differences and produce scores that reflect human preferences?

---

## Slide 4 — Research Goals and Contributions
**3:00–4:00 | Switch to slide 5 at 4:00**

To address this problem, the researchers make three main contributions.

First, they define and investigate fine-grained image aesthetic assessment as a problem involving visually similar images.

Second, they introduce a new benchmark called FGAesthetics.

Unlike conventional aesthetic datasets, it organizes images into series, so the models can be evaluated on subtle differences within the same scene.

Third, they propose a new model called FGAesQ.

This model combines three main techniques: DiffToken, CTAlign, and RankReg.

Each addresses a different part of the problem.

DiffToken helps preserve visual differences, CTAlign provides additional guidance using comparative text, and RankReg teaches the model to predict scores that better reflect relative rankings.

The overall goal is not simply to predict aesthetic scores, but to make those scores more discriminative when comparing similar images.

Let's first look at the dataset.

---

## Slide 5 — The FGAesthetics Dataset
**4:00–5:30 | Switch to slide 6 at 5:30**

The authors created the FGAesthetics dataset specifically for fine-grained aesthetic assessment.

It contains approximately 32,000 images, organized into just over 10,000 image series.

The important feature is the structure of these series.

Instead of collecting unrelated photographs, the researchers group images that show similar visual content.

The dataset contains three categories.

The first is natural photography, with around 13,000 images.

These include burst photographs and sequential video frames, where the same scene is captured multiple times with small differences.

The second category contains AI-generated images.

Here, multiple images are generated from related prompts or generation settings. This creates variations that are semantically similar but visually different.

The third category consists of cropping variations.

Different crops of the same original photograph change the composition while preserving most of the underlying content.

This is especially interesting because cropping alone can change how appealing a photograph looks.

For example, moving the subject away from the center might create a more balanced composition.

By organizing the images this way, the dataset reduces the influence of unrelated scene content.

The model can focus more directly on differences in composition, lighting, framing, and other aesthetic properties.

But collecting similar images is only the first step.

The researchers also needed reliable information about which images people actually prefer.

---

## Slide 6 — Data Collection and Ground Truth Annotation
**5:30–7:00 | Switch to slide 7 at 7:00**

To create the ground-truth rankings, the authors use a process involving data collection, series refinement, human judgments, and rank calibration.

First, they collect candidate image series.

These are then refined using similarity metrics, multimodal models, and human inspection to ensure that the images are suitable for fine-grained comparisons.

Next comes the human annotation process.

Instead of asking people to assign independent numerical aesthetic scores, the researchers ask them to compare images directly.

For example, consider the simplified illustration on this slide.

If Image A is preferred over Image B, and Image C is preferred over Image A, we can establish a ranking where C comes first, followed by A and then B.

The actual annotation process is more systematic.

The authors use multiple human judgments per pair and apply a Bradley–Terry model to aggregate pairwise preferences into relative rankings.

This model represents the probability that one image will be preferred over another.

They also filter ambiguous comparisons, where the preference is too uncertain.

The advantage is that relative comparisons can provide more consistent supervision than asking people to distinguish very small differences using absolute scores.

Ultimately, this gives the researchers a dataset containing not only similar images, but also information about their relative aesthetic quality.

With that dataset, they can train the proposed FGAesQ model.

---

## Slide 7 — FGAesQ Architecture Overview
**7:00–8:30 | Switch to slide 8 at 8:30**

Now let's examine the actual architecture.

The model builds on CLIP's Vision Transformer, specifically ViT-B/16.

As we discussed in class, a Vision Transformer processes images by dividing them into patches and converting these patches into tokens.

These tokens are then processed through transformer layers to produce visual representations.

FGAesQ extends this foundation with three techniques.

The first is DiffToken, which operates during image tokenization.

When a suitable reference image is available, it uses the differences between related images to preserve more detail in distinctive regions.

The second component is CTAlign.

During training, the researchers use comparative text descriptions generated with GPT-4o.

These descriptions provide additional supervision about the aesthetic differences between images.

The third component is RankReg.

This is a ranking-aware training objective that helps the model produce scores consistent with human preferences.

An important distinction is the difference between training and inference.

CTAlign and RankReg provide supervision during training.

During inference, the model uses its visual encoder and regression head to predict aesthetic scores.

It does not need to generate new text descriptions or calculate the training losses.

Now let's look at the three components individually.

---

## Slide 8 — DiffToken: Preserving Visual Differences
**8:30–10:00 | Switch to slide 9 at 10:00**

The first component is DiffToken, or difference-preserved tokenization.

Its purpose is to prevent subtle visual differences from being lost when images are processed by the Vision Transformer.

Imagine two nearly identical photographs.

Most of their regions contain the same information, but perhaps one small area differs because of a moving object, a change in focus, or slightly different framing.

DiffToken tries to preserve more detail in these distinctive regions.

To identify them, it compares corresponding patches from a target image and a reference image.

The paper uses structural similarity, or SSIM, to measure how similar those regions are.

A high SSIM value indicates that two regions are visually similar, while a lower value indicates greater differences.

The model then uses a percentile-based selection criterion to determine which regions should retain greater detail.

The distinctive regions preserve finer visual tokens, while more similar regions can be represented more compactly.

This is useful because Vision Transformers have a computational cost associated with processing image tokens.

Instead of increasing the detail everywhere, DiffToken concentrates the representation on regions where the images differ.

One important limitation is that a visual difference is not automatically an aesthetically meaningful difference.

DiffToken identifies potentially useful regions.

The remaining training objectives help the model learn which differences actually influence aesthetic preference.

---

## Slide 9 — CTAlign: Comparative Text-Assisted Alignment
**10:00–11:30 | Switch to slide 10 at 11:30**

The second component is CTAlign, which stands for Comparative Text-Assisted Alignment.

DiffToken helps preserve visual differences.

However, detecting a difference is not the same as understanding why one image looks better than another.

For example, consider two photographs of the same landscape.

One might have a more balanced composition, while another has distracting objects in the foreground.

Simply knowing that the images are different does not explain which difference matters aesthetically.

To provide additional supervision, the researchers use GPT-4o to generate comparative text descriptions.

These descriptions might mention sharper focus, cleaner framing, or more balanced lighting.

The model then creates two types of representations.

First, it calculates the difference between the visual feature representations of the compared images.

Second, it encodes the comparative text using a CLIP text encoder.

CTAlign uses a cosine-based alignment objective to encourage these representations to capture compatible information.

In simpler terms, the model learns to connect visual differences with language describing those differences.

This helps shape its visual representation during training.

An important detail is that these descriptions are prepared as training supervision.

The text branch is not required during normal inference.

So CTAlign helps the model learn, without requiring a language model whenever we want to score a new photograph.

---

## Slide 10 — RankReg: Rank-Aware Regression
**11:30–13:00 | Switch to slide 11 at 13:00**

The third component is RankReg, or Rank-Aware Regression.

This addresses another limitation of conventional aesthetic scoring.

Typically, regression models are trained to predict a numerical score that matches a target rating.

However, minimizing the numerical prediction error does not explicitly guarantee the correct ordering of similar images.

For example, imagine three photographs with human preferences A, then B, then C.

If a model predicts almost identical scores for all three, a small prediction error could change their ranking.

RankReg introduces relative ranking information into the training objective.

The slide shows two relevant concepts.

The first is the Bradley–Terry preference model.

It converts differences between two predicted scores into a probability that one image is preferred over the other.

If the scores are equal, the predicted preference probability is 50 percent.

As the score difference increases, the model becomes more confident about the ordering.

The second concept is ListMLE.

Instead of considering only individual pairs, it provides a way to optimize the predicted ordering of an entire image series.

The important idea is that the model should learn not only good individual aesthetic scores, but also scores that preserve human-preferred rankings.

The authors also combine fine-grained training with coarse-grained aesthetic supervision from the AVA dataset.

This helps the model improve subtle discrimination without losing its general ability to assess image aesthetics.

---

## Slide 11 — Experimental Setup and Evaluation
**13:00–14:15 | Switch to slide 12 at 14:15**

After introducing the model, the authors evaluate its performance using several benchmarks and metrics.

The main fine-grained benchmark is their new FGAesthetics dataset.

They also use established datasets such as AVA, AADB, ICAA17K, and TAD66K to investigate broader aesthetic assessment and generalization.

They compare FGAesQ against several existing approaches, including CNN-based methods, Vision Transformer models, and multimodal models such as Q-Align.

For fine-grained evaluation, an important metric is pairwise accuracy.

This measures how often the model correctly predicts which of two images is preferred.

The authors also evaluate image series.

Here, series-level metrics measure whether the model identifies preferred images and whether its predicted ranking agrees with the human ranking.

For ranking correlation, they use Spearman's rank correlation.

A value closer to one indicates better agreement.

For general aesthetic scoring, they also report SRCC and PLCC on the AVA benchmark.

These measure the relationship between predicted aesthetic scores and human-provided scores.

Together, these experiments test both fine-grained discrimination and general aesthetic assessment.

---

## Slide 12 — Main Results and Ablation Studies
**14:15–16:15 | Switch to slide 13 at 16:15**

Now let's look at the main experimental results.

On the left side, the chart compares the series-level Spearman ranking correlation of FGAesQ against selected baseline methods.

The evaluation is separated into the three FGAesthetics categories: Natural, AI-generated, and Cropping.

We can see that FGAesQ performs particularly well on the Natural subset, reaching a correlation of approximately 0.729.

The results are also strong on the other categories.

This indicates that the model can produce rankings that agree more closely with human preferences than the selected comparison methods.

However, the differences vary by category, and the best competing method is not always the same.

The right side of the slide shows the ablation study.

An ablation means removing or disabling one part of a model to investigate how much it contributes to the final performance.

The complete FGAesQ model achieves an aggregated fine-grained series score of 0.600.

When DiffToken is removed, this decreases to 0.423.

This is a relatively large drop, suggesting that preserving distinctive visual information is especially important for this task.

When CTAlign is removed, the score decreases to 0.581.

And without RankReg, it decreases to 0.571.

Both components therefore provide additional improvements, although their individual effects are smaller than the effect of DiffToken in this experiment.

Another important observation concerns general aesthetic assessment.

The complete model maintains an SRCC of 0.770 on AVA, indicating that its fine-grained specialization does not necessarily come at the expense of general aesthetic scoring.

Overall, these experiments support the authors' design choices.

The results also illustrate why evaluating individual components is important: a more complex architecture is only useful if those components contribute measurable improvements.

---

## Slide 13 — Contributions and Limitations
**16:15–17:15 | Switch to slide 14 at 17:15**

To summarize, the paper makes two major practical contributions.

The first is FGAesthetics, a benchmark specifically designed for comparing subtle aesthetic differences within image series.

The second is FGAesQ, which combines difference-aware tokenization, comparative text supervision, and ranking-aware training.

The experimental results suggest that these techniques improve fine-grained discrimination while maintaining general aesthetic assessment performance.

However, there are also limitations.

First, the training process depends on human preference annotations.

These can be expensive to collect, and aesthetic preferences are inherently subjective.

Second, the model produces an aesthetic score, but it does not necessarily explain what a photographer should change to improve an image.

Finally, it remains interesting to investigate how well the learned representations transfer to more specialized image domains.

This last point is particularly relevant to my proposed follow-up experiment.

---

## Slide 14 — Follow-Up Experiment: Food Photography
**17:15–18:30 | Switch to slide 15 at 18:30**

For my follow-up experiment, I plan to investigate whether FGAesQ can be adapted to the domain of food photography.

The original model focuses on general image aesthetics, particularly fine-grained comparisons.

Food photography introduces a more specialized context.

For example, the visual appeal of a food photograph can depend on plating, food textures, colors, lighting, and the arrangement of ingredients.

A photograph might have good general composition but still make the food appear less appetizing.

The idea is to investigate whether a small amount of domain-specific training can improve the model's judgments in this area.

I plan to use the Gourmet Photography Dataset, or GPD.

This dataset provides food photographs with positive and negative aesthetic labels.

One important difference is that these are binary labels rather than fine-grained image rankings.

Therefore, my experiment would investigate how well the learned aesthetic representations can transfer to a related but different prediction task.

The main research question is whether lightweight domain adaptation can improve performance compared with the original pretrained model.

---

## Slide 15 — Proposed Experiment and Evaluation
**18:30–19:50 | Switch to slide 16 at 19:50**

The planned experiment consists of four steps.

First, I will evaluate the original pretrained FGAesQ model on the food photography dataset.

This establishes a baseline without domain-specific training.

Second, I will adapt the model.

Rather than retraining the entire architecture, I plan to freeze the pretrained visual backbone and train a small additional scoring head using labeled food photographs.

This keeps the computational effort manageable while testing whether the existing visual representations are useful for the new domain.

Third, I will compare the original and adapted models on a held-out test set.

Because the dataset provides binary labels, suitable evaluation metrics are classification accuracy and ROC-AUC.

I will use separate validation data for selecting thresholds and model settings, so the test set remains independent.

Finally, I will perform an ablation study.

For this, I plan to train the adaptation head using different amounts of training data, such as 25, 50, and 100 percent.

This should help determine whether the adaptation improves performance and how much domain-specific data is needed.

At this stage, the experiment is only proposed, so I cannot make claims about its results.

The goal is to investigate whether specialized training adds value to the original model.

---

## Slide 16 — References and Closing
**19:50–20:00 | End recording**

These are the main references for the paper and the proposed experiment.

That concludes my presentation.

Thank you for listening.