---
title: 'Deep learning meets Kubernetes: Running massively parallel inference pipelines efficiently'
slides: //conferences.oreilly.com/strata-data-ai/stai-ca/public/schedule/detail/80190
video: //conferences.oreilly.com/strata-data-ai/stai-ca/public/schedule/detail/80190
date: 2020-01-11
display_date: March 2020
---

Nearmap captures terabytes of aerial imagery daily. With the introduction of AI capabilities, Nearmap has leveraged Kubernetes to generate AI content based on tens of petabytes of images effectively and efficiently. 

Suneeta Mall walks you through the joys of building and running this system at scale, challenges encountered, the company’s resolution, and future work. Some of the challenges encountered, for instance, have been around provisioning a conditionally stateful fault-tolerant directed acyclic graph (DAG), autoscaling, etcd, networking, GPU orchestration and sharing, and hybrid cluster setup.
Exploiting Kubernetes’s resilient, highly available, extensible, declarative state management capabilities, the company built a fully automated solution that at the core is a DAG—with each node fanning out to achieve said task in shortest time possible. The company uses this architecture for a deep learning inference of billions of high-resolution aerial images on a mix of GPU and CPU compute, primarily driven off the spot pricing scheme. You’ll discover the architecture of this autoscalable solution that exhausted all K80 spot GPUs across all US data centers of AWS for weeks.
Built purely using open source software, this solution is so resilient and elastic that the company has scaled on demand from zero to thousands of compute nodes, crunching through petabytes of images to generate semantic segmentation results with effectively no manual intervention. This system has already produced semantic content on over a million-kilometers-squared area at resolution as high as 5 cm per pixel in just two weeks.

> This talk is upcoming