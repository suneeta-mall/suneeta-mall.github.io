---
title: 'Building More Reliable Data Pipelines for Nearmap Deep Learning Models: An Evolutionary Case Study'
slides: https://www.dropbox.com/scl/fi/c75vjetnxys6q9je3x0n2/Kafka-Summit-APAC-Final-Samanvay-Suneeta.pptx?dl=0&rlkey=cniwh8dcuk3k4kmzlj37vrjl9
video: https://www.dropbox.com/scl/fi/c75vjetnxys6q9je3x0n2/Kafka-Summit-APAC-Final-Samanvay-Suneeta.pptx?dl=0&rlkey=cniwh8dcuk3k4kmzlj37vrjl9
date: 2021-07-28
display_date: July 2021
---


I spoke about our recent experience working with ML pipelines at [Kafka Summit] 2021. The topic was `Building More Reliable Data Pipelines for Nearmap's Deep Learning Models: An Evolutionary Case Study` [link][Talk].

Continual learning using a continually evolving dataset is the norm for the AI team at Nearmap. We have had a software system & data pipelines to facilitate the management of this ever-growing dataset in place for several years of operation. During that time, both our needs & the system have evolved – we improvised and learned from early limitations & challenges. 

One of the biggest challenges of MLOps is building data systems right! Reliable, Fault-tolerant, & continually flowing pipelines are the foundation, with necessary additional capabilities for data quality control, reconciliations, & lineage/tracking.

Based on our learnings, we have rebuilt a new generation of our system (based on Kafka) with one aim – the much-discussed "operation vacation". The aim is to facilitate full automation and zero manual intervention of the system.

In this session, we will go into details of the challenges we encountered, the lessons we learned, what we improved, and lastly; are we on vacation yet?

## Slides and video

Slides can be found [here][slides]. Much to my disappointment, the recording is not available: 

[Kafka Summit]: https://www.kafka-summit.org
[Talk]: https://www.kafka-summit.org/sessions/building-more-reliable-data-pipelines-for-nearmaps-deep-learning-models
[slides]: https://www.dropbox.com/scl/fi/c75vjetnxys6q9je3x0n2/Kafka-Summit-APAC-Final-Samanvay-Suneeta.pptx?dl=0&rlkey=cniwh8dcuk3k4kmzlj37vrjl9
