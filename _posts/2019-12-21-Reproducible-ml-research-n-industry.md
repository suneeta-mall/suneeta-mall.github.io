---
title: Reproducibility in Machine Learning - Research and Industry
tags:
  - machine-learning
  - AI   
date: 2019-12-21
---

---

This is [Part 1] - **Reproducibility in Machine Learning - Research and Industry** of technical blog series titled [Reproducibility in Machine Learning]. [Part 2] & [Part 3] can be found [here][Part 2] & [here][Part 3] respectively.      

---

Machine learning (ML) is an interesting field aimed at solving problems that can not be solved by applying deterministic logic. 
In fact, ML solves problem in logits [0, 1] with probabilities!
ML is highly iterative and fiddly field with much of its **_intelligence_** derived from data upon application of complex mathematics. 
Sometimes, even a slight change such as changing the order of input/data can change the outcome of ML processes drastically.
Actually [xkcd] quite aptly puts it:

>![](/images/xkcd_1838.png)
*Figure 1: Machine Learning explained by XKCD*

This phenomena is explained as **C**hange **A**nything **C**hanges **E**verything a.k.a. CAKE principle coined by Scully et. _al_ 
in their NIPS 2015 paper titled ["_Hidden Technical Debt in Machine Learning Systems_"][scully_2015]. 
CAKE principle highlights that in ML - no input is ever really independent. 

## What is reproducibility in ML

Reproducibility as per Oxford dictionary is defined as something that can be _produced again in the same way_.
> ![](/images/reproducible-oxford.jpeg)
*Figure 2: Reproducible defined*

In ML context, it relates to getting same output on same algorithm, (hyper)parameters, and data on every run. 

To demonstrate, lets take a simple linear regression example (shown below) on [Scikit Diabetes Dataset]. 
A linear regression is all about fitting a line i.e. `Y = a + bX` over data-points represented as X, with b being the 
slope and a being the intercept.   
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()	
diabetes_X = diabetes.data[:, np.newaxis, 9]
xtrain, xtest, ytrain, ytest = train_test_split(diabetes_X, diabetes.target, test_size=0.33)
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
diabetes_y_pred = regr.predict(xtest)
	
# The coefficients	
print(f'Coefficients: {regr.coef_[0]}\n'
      f'Mean squared error: {mean_squared_error(ytest, diabetes_y_pred):.2f}\n'
      f'Variance score: {r2_score(ytest, diabetes_y_pred):.2f}')
# Plot outputs	
plt.scatter(xtest, ytest,  color='green')
plt.plot(xtest, diabetes_y_pred, color='red', linewidth=3)
plt.ylabel('Quantitative measure of diabetes progression')
plt.xlabel('One of six blood serum measurements of patients')
plt.show()
```
Above ML code is NOT reproducible. Every run will give different results: **a)** The data distribution will vary and 
**b)** Obtained slop and intercept will vary. See Figure 3.

> ![](/images/scikit-repro.jpeg)
*Figure 3: Repeated run of above linear regression code produces different results*  

In the above example we are using same dataset, same algorithm, same hyper-parameters. So why are we getting different results? 
Here the method `train_test_split` splits the diabetes dataset into training and test but while doing so, it performs a random shuffle of dataset. 
The seed for this random shuffle is not set here. Because of this every run produces different training dataset distribution. 
Due to this, the regression line slope and intercept are ends up being different. In this simple example, if we were to set random 
state for method `train_test_split` e.g. `random_state=42` then we will have reproducible regression example over 
diabetes dataset. The reproducible version of above regression example is as following:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()	
diabetes_X = diabetes.data[:, np.newaxis, 9]
xtrain, xtest, ytrain, ytest = train_test_split(diabetes_X, diabetes.target, test_size=0.33,
                                                random_state=42)
regr = linear_model.LinearRegression()
regr.fit(xtrain, ytrain)
diabetes_y_pred = regr.predict(xtest)
	
# The coefficients	
print(f'Coefficients: {regr.coef_[0]}\n'
      f'Mean squared error: {mean_squared_error(ytest, diabetes_y_pred):.2f}\n'
      f'Variance score: {r2_score(ytest, diabetes_y_pred):.2f}')
# Plot outputs	
plt.scatter(xtest, ytest,  color='green')
plt.plot(xtest, diabetes_y_pred, color='red', linewidth=3)
plt.ylabel('Quantitative measure of diabetes progression')
plt.xlabel('One of six blood serum measurements of patients')
plt.show()
```
Seeding random state is not the only challenges in writing reproducible ML. In fact, there are several reasons why reproducibility 
in ML is so hard to achieve. But I will go into that a bit later in section `Challenges in realizing reproducible ML`. 
First, question should be  "why reproducibility matters in ML"? 

## Importance of reproducibility in ML

### 1. Understanding, Explaining, Debugging and Reverse Engineering

Reproducibility helps with **_understanding, explaining and debugging_**. Reproducibility is also a crucial means to **_reverse engineering_**.

Machine learning is inherently difficult to explain, understand and also debug. Obtaining different output on subsequent 
run just makes this whole understanding, explaining, debugging thing all the more challenging. How do we ever reverse engineer? 
As it is, understanding and explaining is hard with machine learning. Its increasingly harder with deep learning. 
For over a decade, researches are have been trying to understand what these deep networks learn and yet have not 100% succeeded in doing so.
> ![](/images/deep-net-understand.jpeg) 

From visualizing higher layer features of deep networks [year 2009][Erhan] to activation-atlases i.e. 
what individual neurons in deep network do [year 2017][Olah_viz] to understanding how deep networks decides 
[year 2018][Olah_interpretability] - are all ongoing progressive efforts towards understanding. Meanwhile, explainability 
has morphed into a dedicated field 'Explainable Artificial Intelligence [XAI]. 

### 2. Correctness

> If anything can go wrong, it will <sub>-[Murphy's law]</sub>

Correctness is important as [Murphy's law] rarely fails us. These are some of the examples of great AI failures of our times.  
![](/images/AI-failure.jpeg)
*Figure 4: Example of some of the great AI failures of our times*

Google Photos launched AI capabilities with automatically tagging image. It was found to be tagging [people of dark skin as gorillas][ai_fail_race]. 
Amazon's recruiting software exhibiting [gender bias][ai_fail_aws] or even IBM's Watson giving unsafe recommendation for [cancer treatment][ai_fail_cancer]. 

[//]: # (Perhaps because their classifier model was not trained with enough people of dark skin. But google responded by immediately banning `gorilla`)  
 
ML output should be correct in addition to being explainable. Reproducibility helps achieving correctness through understanding and debugging.   

### 3. Credibility

ML output must be credible. Its not just from fairness, ethical viewpoint but also because they sometimes impact lives (e.g. mortgage approval).
Also, end users of ML output expect answers to verifiable, reliable, unbiased and ethical.
As Lecun said in his [International Solid State Circuit Conference in San Francisco, 2019][lecunn_icc] keynote:
> Good results are not enough, Making them easily reproducible also makes them credible


### 4. Extensibility

Reproducibility in preceding layers are needed to build out and extend. Can we build a building outline model if we cant 
repeatedly generate roof semantics as shown in figure 5? 
![](/images/extensibility.jpeg)
*Figure 5: Extending ML*
  
[//]: # (//TODO: Data generation, correction with GAN) 


## Challenges in realizing reproducible ML

Reproducible ML does not come in easy. A wise man once said:
> When you want something, all the universe conspires in helping you to achieve it. <sub>- [The Alchemist] by Paulo Coelho</sub>

But when it comes to Reproducible ML its quite the contrary. Every single resource and techniques 
(Hardware, Software, Algorithms, Process & Practice, Data) needed to realize ML poses 
some kind of challenge in meeting reproducibility (see figure 6).

<!-- {: .oversized} -->
> ![](/images/reproducible-challenge.jpeg)
*Figure 6: Overview of challenges in reproducibile ML*

### 1. Hardware

### 2. Software

### 3. Algorithm

### 4. Process & Practice

### 5. Data


## Replicability and 

The research community is quite divided when it comes to defining reproducibility and often mixes it up with replicability 
> ![](/images/replicable.jpeg)
*Figure 4: Replicability defined*




[Reproducibility in Machine Learning]: /2019/12/20/Reproducibility-in-machine-learning.html
[Part 1]: /2019/12/21/Reproducible-ml-research-n-industry.html
[Part 2]: /2019/12/22/Reproducible-ml-tensorflow.html
[Part 3]: /2019/12/23/Reproducible-ml-pipeline-k8s.html
[xkcd]: //xkcd.com/1838
[Erhan]: //www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/247
[Olah_viz]: //distill.pub/2017/feature-visualization
[Olah_interpretability]: //distill.pub/2018/building-blocks/
[scully_2015]: //papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf
[Scikit Diabetes Dataset]: //scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
[XAI]: //arxiv.org/abs/1910.10045
[lecunn_icc]: //twitter.com/ylecun/status/1097532314614034433
[Murphy's law]: //en.wikipedia.org/wiki/Murphy%27s_law
[ai_fail_race]: //twitter.com/jackyalcine/status/615329515909156865
[ai_fail_cancer]: //www.theverge.com/2018/7/26/17619382/ibms-watson-cancer-ai-healthcare-science
[ai_fail_aws]: //medium.com/syncedreview/2018-in-review-10-ai-failures-c18faadf5983
[The Alchemist]: //www.amazon.com/Alchemist-Paulo-Coelho/dp/0061122416

