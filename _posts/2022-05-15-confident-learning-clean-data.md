---
title: "Confident Learning: Err, Did you say your data is clean?"
tags:
  - Machine-learning
  - AI
  - Deep-Learning
  - Data
  - Data-Centric-AI
  - Confident-Learning
date: 2022-05-16
---


What makes deep-learning so great, despite what you may hear about, is data! There is an old saying, that sums it up pretty well:

> The model is only as good as the <span style="color:red">data</span>!
<br/>


Which brings us to the real question, exactly how good is your data? Collecting ground truth/training datasets is incredibly expensive, laborious, and time-consuming. The process of labelling involves searching for the object of interest and applying prior knowledge and heuristics to come to a final decision. A decision to represent if the object (of interest) is present and if so, annotate it to provide extra information like bounding box, segmentation mask. 

- [Reasons for labelling errors](#reasons-for-labelling-errors)
- [Labelling efficiency tricks that increase the risk of labelling errors](#labelling-efficiency-tricks-that-increase-the-risk-of-labelling-errors)
- [Ok! Ok! Errors in labels are imperative](#ok-ok-errors-in-labels-are-imperative)
- [Exactly, what is Confident learning?](#exactly-what-is-confident-learning)
  - [Fundamentals](#fundamentals)
  - [Limitations of CL](#limitations-of-cl)
- [let us try this cool thing out](#let-us-try-this-cool-thing-out)
  - [Model info](#model-info)
  - [Exploration in noise using cleanlab](#exploration-in-noise-using-cleanlab)  

# Reasons for labelling errors

There are many reasons why errors in the labels may be present. One broad class for such errors is tooling/software bugs. These can be controlled and managed using good software best practices like tests covering both software and the data. The other class of errors is the one where the mistakes are coming from the labellers themselves. These are incredibly harder issues to track. Because they are the oracle in this process of deep learning after all! If we don't trust them, then who do we trust?

There is actually very interesting work by [Rebecca Crowley] where she provides a detailed chart of a range of reasons why an object (of interest) may be missed while searching in a scene  or also why a wrong final decision may be made by them. Some directly impacting labelling in my view are:

1. **Search Satisficing**: The tendency to call off a search once something has been found, leading to premature stopping, thus increasing the chance of missing annotations. This more applies to scenarios where more than one annotation is needed. For example, multi-label or segmentation annotation (dog and pen are in the image but labeller only annotates for the dog and does not spend enough time to spot the pen and capture in labels).
2. **Overconfidence & Under-confidence**: This is sort of self-explanatory. Relates to the belief in one’s own performance, with extremes representing opposite ends of a spectrum of feeling-of-knowing.
3. **Availability**: If something is frequently occurring or rarely occurring, then there is an implicit bias in annotating a wrongly. This is particularly true for challenging labelling tasks. For instance, if the cancer prevalence rate in a location is 0.01%, then labeller, when labelling a not-so-straightforward case is more likely to make a non-cancer than cancer.
4. **Anchoring/Confirmation bias**: When labeller makes an early pre-emptive decision of what this task results are before spending enough time collecting info from all data and ends up deciding unfairly.
5. **Gambler’s Fallacy**: Tendency to deviate in opposite directions in repeated pattern has been showing up in a sequence - 10 images are of cat, more likely to believe the next one will not be a cat if a bunch of previous cases has been positive for a cat!
6. Amongst all these, **Cognitive Overload** is also a valid and fair reason for errors in labels.


# Labelling efficiency tricks that increase the risk of labelling errors

There are some tricks out there to optimise the labelling funnel. While some focus on optimisation through labelling experience such as [Fast Interactive Object Annotation], other techniques focus on using auto-labelling techniques to reduce the labelling burden in a bit to aid the labellers. Tesla has a very powerful mechanism to realize auto-labelling at scale as talked about in [2021 CVPR by Karpathy][tesla_cvpr_2021]. They sure have the advantage of having feedback (not just the event that's worth labelling but also what did the driver do or if that led to a mishap). It's an advantage that not all deep-learning practicing organizations have! Impressive nonetheless! 
Then we also have the weakly supervised class of training regimes that I won't go into detail about (perhaps a topic for another day)!

The thing is, the cleverer tricks you employ to optimise this process, the higher the chances of error in your dataset. For instance, using the model in the loop for labelling is being increasingly used to optimize the labelling process ([as detailed in this presentation][she_builds_talk]), as an auto-labelling/pre-label trick, wherein predicted labels are shown to the labellers and labeller only fine-tunes the annotation instead of annotating from the get-go. This process has two main challenges:

a) It may introduce a whole gamut of bias in labels [as discussed above in (Reasons for labelling errors](#reasons-for-labelling-errors).

b) If the model is not on par with human labellers, the labour, and boredom of correcting a garbage prior label increase the risk of errors in the dataset. This is a classic case of cognitive overload.

If this example was not enough, let's take a case of auto labelling [using ontology/knowledge graph][kgc_onto_talk]. Here, the risk of error propagation is too high if the knowledge encoded in the ontology/knowledge graph is biased. For example, if it's an ocean water body it cants be a swimming pool. Because well, contrary to common knowledge ocean pools do exist. Or if it's a house it's not a waterbody - because you know lake houses do exist!

>![](/images/data-centric-ai/cleanlab/47f76ecac3e3bcc2b5b21e30ea10d336.jpeg)
(Image is taken from the internet! @credit: unknown)

# Ok! Ok! Errors in labels are imperative

I diagress, but point I want to underscore here is:
> I am <span style="color:red">not perfect</span>; I am only <span style="color:red">human</span>

Evidently, there is *an estimated average of at least 3.3% errors across the 10 popular datasets, where for example label errors comprise at least 6% of the ImageNet validation set* [1][benchmark_northcutt].

Assuming, human labellers will produce a perfectly clean dataset is, well, overreaching to say the least. If one really cares, one can employ multiple labellers to reduce the error by removing the noise through consensus. This is a great technique to produce a high-quality dataset but it's also many times more expensive and slow thus impractical as standard modus-operandi. Besides being expensive, this does not guarantee a clean dataset either. This is evident from [Northcutt's NeurIPS 2021][benchmark_northcutt] work on analysing errors in the test set of popular datasets that reported order of hundred samples across popular datasets where an agreement could not be reached on true ground truth despite looking at collating outcomes from labellers (see table 2 in the paper for reference).


For the last few years, I have been using model-in-loop to find samples where there are disagreements in the dataset with the models. Some of the other techniques that I have found useful are leveraging loss functions, as called out by Andrej Karpathy! More recently, I have seen a huge benefit of deploying [ontology-based voilations][kgc_onto_talk] to find samples that are either a) labelling errors or b) extreme edge cases that we were not aware of. 


<blockquote class="twitter-tweet"><p lang="en" dir="ltr">When you sort your dataset descending by loss you are guaranteed to find something unexpected, strange and helpful.</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/1311884485676294151?ref_src=twsrc%5Etfw">October 2, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 

However, I realized I have been living in oblivion when I came across project [cleanlab] from [Northcutt]'s group and started digging in a bunch of literature around this exact space! I was excited to uncover all the literature that proposed the tricks and tips I have been using to date without being aware of them. Well no more!!
In the following sections, I will cover what I have learned reading through this literature.


# Exactly, what is Confident learning?

> All models are wrong, but some are <span style="color:red">useful</span>!

[Confident learning][confident_learning] (CL) is all about using all the useful information we have at hand to find noise in the dataset and improve the quality of the dataset. It's about using one oracle (the labellers) and testing it using another oracle in the build i.e. the model! 
At a very high level, this is exactly what we have been more generally calling model-in-the-loop!

> Learning exists in the context of data, yet notions of confidence typically focus on model predictions, not label quality!

[Confident learning][confident_learning] (CL) is a class of learning where the focus is to learn well despite some noise in the dataset. This is achieved by accurately and directly characterizing the uncertainty of label noise in the data. The foundation CL depends on is that `Label noise is class-conditional, depending only on the latent true class, not the data` [1][confident_learning]. For instance, `leopard` is likely to be mistakenly labelled as `jaguar`.  This is a very strong argument and in practice untrue. I can argue that the scene (ie the data) has implications for labellers decisions. For instance, a dingy sailing in water is less likely to be labelled as a car if it's in water. In other words, would not a dingy being transported in a lorry is more likely to be labelled as a car than when it was sailing on the water? so data and context do matters!   
This assumption is a classic case of, `in statistics any assumption is fair if you can solve for x!`. Now that we have taken a dig at this, it's fair to say that this assumption is somewhat true even if not entirely. 

Another assumption CL makes is that rate of error in the dataset is < 1/2. This is a fair assumption and is coming from [Angluin & Laird]'s proposed method for the class conditional noise process which is used in CL to calculate the joint distribution of noisy and true labels.

In many ways [Confident learning] is all about using a reasonably well performant model in the loop to estimate the errors in the dataset. First, the model's prediction is obtained then, using a class-specific threshold setting confident joint distribution matrix is obtained; which is then normalized to obtain an estimation of the error matrix. This estimated error matrix then builds the foundation for dataset pruning, counting, and ranking samples in the dataset.

> Estimating the joint distribution is challenging as it requires disambiguation of epistemic uncertainty (model-predicted probabilities) from aleatoric uncertainty (noisy labels) but useful because its marginals yield important statistics used in the literature, including latent noise transition rates latent prior of uncorrupted labels, and inverse noise rates


>![](/images/data-centric-ai/cleanlab/cl.jpg)
[Confident learning][confident_learning]  in pictures 

The role of class-specific threshold is useful to handle variations in model performance for each class given the assumption here is that the models used in generating probabilities are uncalibrated. This technique is also helpful to handle collisions in predicted labels. 

What's great about [Confident learning][confident_learning] and [cleanlab] (its python implementation of it) is that while it does not propose any groundbreaking algorithm, it has done something that is rarely done. Bring together antecedent works into a very good technical framework that is so powerful that it rightfully questions major datasets that shape the evolution of deep learning. Their work on [analysis][benchmark_northcutt] of 10 popular datasets including MNIST is well appreciated. This is also a reminder that we are massively overfitting the entire deep learning landscape to datasets like MNIST, ImageNet, as they are pretty much, must use dataset to benchmark and qualify bigger and better algorithms and that, they have at least 3.3% errors if not 20% (as estimated by [Northcutt's group][benchmark_northcutt])!


This approach was used in side by side comparison where samples were pruned either randomly (shown in orange in below fig) or more strategically via CL to remove noisy samples only (shown in blue in below fig). The accuracy results as shown below. It's clear that CL is doing better than random prune!
>![](/images/data-centric-ai/cleanlab/cl_result.jpg)
Borrowed from [Confident learning][confident_learning]


Following are an example of some of the samples that CL flags as noisy/erroneous, also including the edge cases where a) neither were true and b) Its either the CL suggestion or provided label but unclear which of the two are correct (non-agreement):

>![](/images/data-centric-ai/cleanlab/cl_results_image.jpg)
Examples of samples corrected using [Confident learning][benchmark_northcutt]


CL is obviously not 100% accurate. At the best, it is an effort to estimate noise using an epistemic uncertainty predictor [the model]. Having said that these examples it fails on are quite challenging as well.

>![](/images/data-centric-ai/cleanlab/fail_cl.jpg)
Examples of samples corrected using [Confident learning][benchmark_northcutt]



## Fundamentals 

Let's look at the fundamentals CL builds on and dig in a bit on the antecedent and related works that formulate the foundation upon which [Confident learning][confident_learning]([cleanlab]) is built!

1. As discussed above already, is class conditional noise proposed by [Angluin & Laird] is the main concept utilized by CL. 

2. Use of weighted loss functions to estimate the probability of misclassification [1][Natarajan_neurips_2013] is quite relevant to the field of CL, although it's not directly used in the CL technique as proposed by [Confident learning][confident_learning] framework paper. 

3. Use of iterative noise cross-validation [INCV] techniques that utilize a model-in-the-loop approach to finding noisy samples as done by [Chen] *et. al*. This algorithm is shown below:

    >![](/images/data-centric-ai/cleanlab/INCV.jpg)
    [Chen]'s INCV algorithm

4. CL does not directly use [MentorNet]. However, it is very relatable work that builds on a data-driven training approach. In simplistic terms, there is a teacher network that builds a curriculum of easy to hard samples (derived using loss measures) and the student is trained off this curricum.

    >![](/images/data-centric-ai/cleanlab/M-Net.jpg)
    [MentorNet] architecture


5. There are other variations to data-driven teaching, one noticeable example is [Co-teaching] which looks to be teaching each other in pair and passing the non-noisy samples (calculated based on loss measures) to each other during learning. The main issue [Co-teaching] tries to solve is memorization effects that are present in [MentorNet] (aka M-Net] but not in [Co-teaching] due to data share.

    >![](/images/data-centric-ai/cleanlab/m-d-co-teach.jpg)
    [Co-teaching] approach in contrast with [MentorNet] aka M-Net

6. The understanding that small losses are a good indicator of useful samples and highly likely correct samples whereas jumpy, high loss producing samples are, well, interesting! They can be extreme edge cases or out-of-distribution samples or also can indicate wrongness. This obviously only holds for reasonably performant models with abilities at least better than random chance if not more!

## Limitations of CL

One thing that CL entirely ignores about label errors is when one or more true label is entirely missed. This is more likely to happen in multi-label settings and even more complex labelling setting like segmentation or detection than in multi-class classifications. For example, if there is a TV and water bottle in an image and the only annotation present is for the TV, and water bottle is missed entirely! This is currently not modelled in CL as relies on building a pairwise class-conditional distribution between the given label and the true label. If the given label was a cat but the actual label was the dog for example. The framework itself does not allow it to model for the missing label when a true label is present. 


Pairwise class-conditional distribution as proposed in CL also limits its use in multi-label settings. Explicitly when multiple labels can co-exist on the same data. For example, both the roof and swimming pool can be present in the image and they are not necessarily exclusive. This limitation comes from pairwise (single class gives vs single class predicted) modelling. This is different from say [Stanford Car Dataset] where make and model are predicted as multiple labels but they are exclusive ie a vehicle can be ute or hatchback but that is exclusive to it being made by Toyota or Volvo. Exclusivity in this case allows for modelling Pairwise class-conditional distribution. These are more multi-label multi-class modelled using multi-headed networks. True multi-label cant be modelled like these pairwise joint distributions. They can perhaps be modelled using more complex joint distribution formulations but that would not scale very well. [Confident learning][confident_learning] as it stands currently is a computationally expensive algorithm with an order of complexity being O(m2 + nm). 



# let us try this cool thing out

I just had to give [cleanlab] a crack and given the uncertainty around multi-label, I did a quick spike to look at label-noise handling on [MLRSNet](https://paperswithcode.com/dataset/mlrsnet) dataset. 

The source-code and notebooks for this project is [label-noise](https://github.com/suneeta-mall/label-noise). This spike is built using ResNet as multi-label image classifier predicting 6 classes that can co-exists at the same time. These classes are `['airplane', 'airport', 'buildings', 'cars', 'runway', 'trees'] ` and derived off [MLRSNet](https://paperswithcode.com/dataset/mlrsnet) subset - i.e. only using `airplane` and `airport`.

[This notebook](https://github.com/suneeta-mall/label_noise/blob/master/label_noise_notebook.ipynb) checks how well [cleanlab] performs in detecting label-noise, identifies out of distribution samples, weird samples, and also errors!

## Model info

Train and validation loss from the model trained using [label-noise](https://github.com/suneeta-mall/label-noise) for 19 epochs. 
![](/images/data-centric-ai/cleanlab/model_measures.jpg)


## Exploration in noise using cleanlab

As shown in [this notebook](https://github.com/suneeta-mall/label_noise/blob/master/label_noise_notebook.ipynb), the filter approach provided a list of samples that were deemed noisy. This basically flagged 38% of out-of-samples as noisy/erroneous. 

Ranking provided an order in the samples on a 0-1 scale for label quality, the higher the number, the better quality the sample. I looked in `get_self_confidence_for_each_label` approach for out-of-distribution (OOD) samples and also entropy-based ordering. The distribution for each is as follows:

![](/images/data-centric-ai/cleanlab/rank_entropy.jpg)
>Confidance order of entropy-based ordering

![](/images/data-centric-ai/cleanlab/rank_self_confidance.jpg)
>Confidance order of self_confidence ordering


The samples picked as lowest quality or confidence are indeed rightfully chosen. There is an airplane that was missed in the image (shown in the highlighted overlay) and also another sample is clearly OOD!
![](/images/data-centric-ai/cleanlab/Errors_flagged_in_ranking.jpg)

What's more interesting is all three methods of filtering, ranked by self-confidence and entropy all flagged these two samples!
So we increase the threshold, and while there are false positives (for noise) there is some good example of errors.



| False Positive | True Positive |
|----------------|---------------|
|  ![](/images/data-centric-ai/cleanlab/fp_cl.jpg)              |     ![](/images/data-centric-ai/cleanlab/Increasing_threshold_error.jpg)          | 


I am not sure if I am following how to interpret the [pair-wise count for multi-label](https://github.com/cleanlab/cleanlab/issues/263), so that's left for another day!

Happy cleaning!!




[she_builds_talk]: https://suneeta-mall.github.io/talks/She_Builds_on_AWS_2020.html
[kgc_onto_talk]: https://suneeta-mall.github.io/talks/KGC_NY_2022.html
[Rebecca Crowley]: https://www.researchgate.net/publication/225041737_Automated_detection_of_heuristics_and_biases_among_pathologists_in_a_computer-based_system
[benchmark_northcutt]: https://arxiv.org/abs/2103.14749
[confident_learning]: https://arxiv.org/abs/1911.00068
[cleanlab]: https://github.com/cleanlab/cleanlab
[Northcutt]: https://twitter.com/cgnorthcutt
[Angluin & Laird]: http://homepages.math.uic.edu/~lreyzin/papers/angluin88b.pdf
[Fast Interactive Object Annotation]: https://arxiv.org/abs/1903.06874
[tesla_cvpr_2021]: https://www.youtube.com/watch?v=a510m7s_SVI
[Stanford Car Dataset]: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
[MentorNet]: https://arxiv.org/abs/1712.05055
[Natarajan_neurips_2013]: https://proceedings.neurips.cc/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf
[Chen]: https://arxiv.org/abs/1905.05040
[Co-teaching]: https://arxiv.org/abs/1804.06872