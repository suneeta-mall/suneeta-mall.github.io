---
title:  Running Massively Parallel Deep-learning Inference Pipelines on Kubernetes
slides: https://static.sched.com/hosted_files/k8sforumsydney19/ad/Offline_Inference_at_Scale_KubeSydForum_2019.pdf
video: https://www.youtube.com/watch?v=WSMYfAEdAe4
date: 2019-12-12
display_date: Dec 2019
---

Nearmap captures terabytes of aerial imagery daily. With the introduction of artificial intelligence (AI) capabilities, Nearmap has leveraged Kubernetes to generate AI content based on tens of petabytes of images effectively and efficiently.

This talk covers how using Kubernetes as the backbone of our AI infrastructure, allowed us to build a fully automated deep-learning inferential pipeline that despite not being embarrassingly parallel is actually massively parallel. This talk explains the architecture of this auto-scalable solution that has exhausted all K80 spot GPUs across all US data centres of AWS for weeks. This system has already produced semantic content on over a million km2 area at resolution as high as 5cm/pixel in just 2 weeks. In this talk, you will learn about the joys of building and running this system at scale, challenges encountered, their resolution, & future work.

## Slides and video

Slides can be found [here][slides] and [video]: 
[![Talk](http://img.youtube.com/vi/WSMYfAEdAe4/0.jpg)](https://www.youtube.com/watch?v=WSMYfAEdAe4)


[slides]: https://static.sched.com/hosted_files/k8sforumsydney19/ad/Offline_Inference_at_Scale_KubeSydForum_2019.pdf
[video]: https://www.youtube.com/watch?v=WSMYfAEdAe4
