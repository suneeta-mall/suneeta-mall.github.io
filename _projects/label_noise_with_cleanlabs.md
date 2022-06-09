---
title: "Label noise analysis using cleanlab"
source: //github.com/suneeta-mall/label_noise
demo: //github.com/suneeta-mall/label_noise/blob/master/label_noise_notebook.ipynb
date: 2022-05-16
display_date: May 2022
---

Errors in datasets are unavoidable - not just because Murphy's Law always holds but because it's a complicated process involving humans, perception, cognition, and software and systems! So, how do we manage them? It's something that I have been thinking a lot about lately. I, recently came across "Confidant learning", a proposed technique by [Curtis Northcutt] and his group.  It piqued my interested so I spent sometime exploring [Cleanlab]. 

If this is something that keeps you up at night too, then here's my initial [notes][blog] focussing specifically on classification problem space. This article was also published in [TDS]. [label_noise] itself is a spike project that using PyTorch, PyTorch lightening and Cleanlab to find errors in MLSRNet dataset. The git repository where all details are stored can be found [here][label_noise].


[blog]: https://suneeta-mall.github.io/2022/05/16/confident-learning-clean-data.html
[label_noise]: //github.com/suneeta-mall/label_noise
[Cleanlab]: //github.com/cleanlab/cleanlab
[Curtis Northcutt]: https://twitter.com/cgnorthcutt
[TDS]: https://towardsdatascience.com/confident-learning-err-did-you-say-your-data-is-clean-ef2597903328
