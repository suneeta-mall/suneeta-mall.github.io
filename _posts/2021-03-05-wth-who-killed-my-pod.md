---
title: WTH! Who killed my pod - Whodunit?
tags:
  - Kubernetes
  - OOM
date: 2021-03-05
---

Few days ago, I deployed a brand new application onto a self managed Kubernetes cluster (hereafter referred as Kube). 
Suffice to say, all hell broke loose. The pods were getting `OOMKilled` with error code 137 left and right! 

Now, I know a thing or two about Kubernetes<sup>[1],[2]</sup>. I am not a total Kube noob!  
But, I could not figure out what the fudge was going on actually! 
Besides, this app has been thoroughly tested and profiled and ran fine on bare metal and virtual environments.

So this was me, a few days ago!.
<!-- {: .oversized} -->
> ![](/images/oom/4201968f94aacab1c0190d9688daba00-sticker.jpg)

This sparked a massive hunt for the culprit, and some interesting insights were discovered. Worth noting, similar investigating
has also been done on by [Line Corp][line-eng-qos] in their excellent blog however I have a different story to tell!
 
In this writeup, 
I am going to talk about this particular incident and insights I have uncovered about both Kube and Linux kernels.


## Context of the app
The app runs some intensive [numpy] and [Tensorflow] computations to produce some artifacts and associative metadata.
The workloads are more memory intensive as they operate on rich multi-media content. 
Other gory details besides _resource requirements_ of the app is  irrelevant for this discussion. 

The average resource requirement, for this app, is very fluctuating yet predictable (in a given range). 
At least so we thought looking at our metrics:
    
<!-- {: .oversized} -->
> ![](/images/oom/avg-resource-requirement.jpg)
*Figure 2: Average resource requirements of the app when run on VMs or bare metal*

I hear you, the resource utilization is not following a zero gradient line (fig 2)! 
It would be awesome to have a constant non flapping resource requirement needs - so clearly there is some work 
that needs to happen on the app here. Having said that, its an absolutely acceptable and supported workload. 

Ok, so the app was deployed and now, we will look at the line of investigation:

## App's on Kube: day 1 

The provisioned app pods started to get killed as frequently as every 20 mins or more with error code 137 and reason `OOMKilled`. 

> ![](/images/oom/Joys%20of%20being%20killed!.jpg)
*Figure 3: Killer's on the loose! - Whodunit?*


Let me explain a few things about the failure first:
1. `Error code 137` indicates that the container process received the SIGKILL and thus was killed by the OS kernel. 
    SIGKILL on Kube can only be produced using one of following means:
    
    1.1. Manually (human): Triggering CTRL+C or using other means of manually sending SIGKILL or even manually killing process.
    
    1.2. Container Runtime/Interface: `Kubelet` the process running on host machine that manages running kube workload is `the power that be` for containers. 
    It communicates through container runtime to manage container lifecycle. It can kill and almost always kills badly behaving pods!  

    > ![](/images/oom/CRI.png)
    *Figure 4: Container runtime interface. Image Credit: [Ian Lewis]! Borrowed from his 4 part container runtime series [container runtime] that I highly recommend watching*

    1.3. OS kernel: The OS kernel is responsible for life cycle of processes running on host. 
    It is `the mighty power that be` for all the processes on host including container process and its children.
    It can also kill and almost always kills badly behaving processes!
    

2. `OOMKilled` represent a kill event (SIGKILL) triggered to a process because someone _in-charge_ suspected of the 
process to be culprit of a memory surge that may lead to an out of memory event. This is a safeguard mechanism to avoid 
system level failure and to nip mischieve in the bud.  

`Takeaway 1`: Either Container Runtime/Interface or OS Kernel killed my process because supposedly it was misbehaving and causing out of memory issue! 
Essentially, I am ruling out manual because that was simply not the case!

### Deep-dive into factors at play here

1. [control groups][cgroups] are a Linux kernel feature that allow processes to be organized into hierarchical groups whose 
usage of various types of resources (memory, CPU, and so on) can then be limited and monitored. The cgroup interface is 
provided through a pseudo-filesystem called cgroupfs. You may have heard about `/sys/fs/cgroup/`!  
[Liz Rice] did an excellent demonstration of [what it means to run container and how they work][container from scratch] that I highly recommend
going through. Dont forget playing with the [demo code][source]. It gives a foundational understanding of cgroup' role in all things containers.

2. `Kubelet` not only interfaces container runtime but also has `cAdvisor` [**C**ontainer **Advisor**][Container Advisor] integrated within. 
Note `kubelet` is service running on host and it operate at host level not pod. 
With `cAdvisor` it captures resource utilization, statistics about [control group][cgroups] of all containers processes on host.

3. Kubernetes manages the resource for containers using `cgroup` that guarantees resource isolation and restrictions. 
Kube can allocate X amount of resources to a container and allow the resources to grow until a pre-existing limit is reached or no more is left on host to use.
Kube provides these [requests and limits] semantic on containers which are used to enforce the said limit on process hierarchy for each container via cgroup.
Now, the `limit`is not hard cut off. As documented in google's blog of [best practices resource requests and limits], there are two types of resources:
    1. _Compressible_ resources: When resource limit is reached, kube will throttle container i.e. start to restricted the usage but wont actually terminate the container. 
        CPU is considered as compressible resource.
    1. _Incompressible_ resources: When a limit for this type of resource is reached, the highest usage process within the cgroup hierarchy will be killed.
        Memory is incompressible resource. 

    `Takeaway 2`: Its not the CPU limit, but memory limit that we need to focus on.

4. Kubernetes classifies pod into three categories based on quality of service (QoS) they provide:
    
    4.1 _Guaranteed_ pods are those who's resource request and limit are just the same. These are the best kind of workload 
    from Kube's view point as they are easier to allocate and plan for resource-wise. These pods guaranteed to not be killed until they exceed their limits.
    > ![](/images/oom/qos-guranteed.jpg)
    *Figure 5: Guaranteed QoS pod example*

    4.2 _Best-Effort_ pods are those where no resource requirements are specified. These are the lowest priority pods and 
    the first to get killed if the system runs out of memory. 
    > ![](/images/oom/qos-best%20effort.jpg)
    *Figure 6: Best-Effort QoS pod example*

    4.3 _Burstable_ pods are those who's resource request and limit are defined in a range (fig 6), with limit treated as max if undefined. 
    These are the kind of workload that are more likely to be killed when the host system is under load and they exceed their requests and no Best-Effort pods exist. 
    > ![](/images/oom/qos-bustable.jpg)
    *Figure 7: Burstable QoS pod example*


    ```
    So can Kube over-commit? 
    If yes, would it always be on the compressible resources? 
    ```
    
    Yes, Kube can overcommit. The pod limits are allowed to be higher than requests. Its possible that the sum of all limits can be higher than 
    total node capacity. Its possible to overcommit both compressible and incompressible resources. This is pictorially explained [here][sysdig's].
    Infact, with Kube its also possible to not only vertically overcommit but also horizontally (at cluster level) overcommit. 
    Horizontal overcommit are nicer as they can trigger autoscaler like event to scale out.   
    


### So why the pods were killed App's on Kube: day 1 diagnosis

The app was initially deployed with `Burstable` QoS with Memory requirements set at request: 4Gi, limit: 7Gi, and 
CPU set at 2 for both request, limit (see fig 2). The nodes were AWS `r5.2xlarge` type with 8 CPU, 64GB RAM.
Other than Kube system components and the app, nothing else was deployed on these nodes. 

So, Kube could have only deployed 3 app pod per `r5.2xlarge` nodes (due to CPU request). This means, 43GB (=64-7*3) of RAM 
was lying around singing hakuna matata! What a waste! Sure but lets not digress!
So why the OOMKill? `¯\_(ツ)_/¯`

Noteworthy observation:
    - Node monitoring tells us that is running healthy and has plenty of resources at disposal. 
    - the pod is still OOMKilled but not all app pods on node, just one is killed.

I am still clueless. So, caving in, I decided to use up this extra memory floating around and beef up the nodes a bit more 
and buy more time to do proper investigation. 
Now, the apps are redeployed again with RAM request 4Gi, limit: 30Gi (leaving 4GB for other misc system components).

Did that ameliorate the problem - no! Of-course, I am being silly about this, I should be making it guaranteed to have better 
chance of avoiding OOMKill. 


## App's on Kube: day 2

So, my apps are running with guaranteed QoS with 30GB of RAM as required/limit. Node still seem healthy and shows no sign of duress.  

Hows the app doing with new revised configuration: `still getting OOMKilled with 137 error code left and right!` 

> ![](/images/oom/34b8525b2cff89f7f25f2f70d62c5014-sticker.png)

Meanwhile, we uncovered random memory surges in some pods (see figure 8). These surges occurred very rarely and did not match
to duration of out of memory kill event. In fact the frequency of OOM was much higher these memory surges. 

> ![](/images/oom/memory-spike.jpg)
*Figure 8: The notorious spike of memory use on pod*

While, these surges are worth investigating, they are still within the request/limit range (28.x Gi Survey on 30Gi request).
So they still dont justify the OOM event.

### Whats log telling us

Based on Takeaway 1 & 2, we look at who is firing the kill signal. #Whodunit

#### Kube events for pod and other higher level abstractions
Investigating, on Kube `Events` there is no record or any OOMKill or any event signalling anything malicious.
```bash
kubectl describe pod <my pod>
kubectl describe deploy <my pod>
```
In fact, according to my kube, event stream `kubectl get events` is all healthy and there is nothing to see, nothing to worry there!
It shows that containers are clearly being restart but it seem to be not capturing any adverse event and bringing it back up to keep
to desired declared state on attached replicaset. 
```text
26m         Normal   Created   pod/myapp   Created container planck
26m         Normal   Started   pod/myapp   Started container planck
26m         Normal   Pulled    pod/myapp   Container image "blahblahblah" already present on machine
```

#### What is the CRI and kubelet doing?

Looking at system journal, there is nothing noteworthy recorded for OOM. 
1. Nothing is logged for `Out of memory` (command reference `journalctl -u kubelet | grep -i "Out of memory"`)
2. Only log I see for shorter term `oom` (cmd reference `journalctl -u kubelet | grep -i "oom"` is info level log of kubelet startup record.  
```text
kubelet[2130]: I0309 04:52:13.990735    2130 flags.go:33] FLAG: --oom-score-adj="-999"
kubelet[2130]: I0309 04:52:15.416807    2130 docker_service.go:258] Docker Info: &{ID:XF74:2JFW:UOE4:QI7X:TXQU:RJLG:E7FC:K4K3:IUTM:MGFW:W2GM:Z6AC Containers:0 ContainersRunning:0 ContainersPaused:0 ContainersStopped:0 Images:0 Driver:overlay2 DriverStatus:[[Backing Filesystem extfs] [Supports d_type true] [Native Overlay Diff true]] SystemStatus:[] Plugins:{Volume:[local] Network:[bridge host macvlan null overlay] Authorization:[] Log:[awslogs fluentd gcplogs gelf journald json-file logentries splunk syslog]} MemoryLimit:true SwapLimit:false KernelMemory:true KernelMemoryTCP:false CPUCfsPeriod:true CPUCfsQuota:true CPUShares:true CPUSet:true PidsLimit:false IPv4Forwarding:true BridgeNfIptables:true BridgeNfIP6tables:true Debug:false NFd:23 OomKillDisable:true NGoroutines:44 SystemTime:2021-03-09T04:52:15.411198727Z LoggingDriver:json-file CgroupDriver:cgroupfs NEventsListener:0 KernelVersion:4.9.0-14-amd64 OperatingSystem:Debian GNU/Linux 9 (stretch) OSType:linux Architecture:x86_64 IndexServerAddress:https://index.docker.io/v1/ RegistryConfig:0xc00062c0e0 NCPU:16 MemTotal:133666107392 GenericResources:[] DockerRootDir:/var/lib/docker HTTPProxy: HTTPSProxy: NoProxy: Name:ip-172-30-36-152 Labels:[] ExperimentalBuild:false ServerVersion:18.06.3-ce ClusterStore: ClusterAdvertise: Runtimes:map[runc:{Path:docker-runc Args:[]}] DefaultRuntime:runc Swarm:{NodeID: NodeAddr: LocalNodeState:inactive ControlAvailable:false Error: RemoteManagers:[] Nodes:0 Managers:0 Cluster:<nil> Warnings:[]} LiveRestoreEnabled:false Isolation: InitBinary:docker-init ContainerdCommit:{ID:468a545b9edcd5932818eb9de8e72413e616e86e Expected:468a545b9edcd5932818eb9de8e72413e616e86e} RuncCommit:{ID:a592beb5bc4c4092b1b1bac971afed27687340c5 Expected:a592beb5bc4c4092b1b1bac971afed27687340c5} InitCommit:{ID:fec3683 Expected:fec3683} SecurityOptions:[name=seccomp,profile=default] ProductLicense: Warnings:[]}
kubelet[2130]: I0309 04:52:15.437879    2130 manager.go:1159] Started watching for new ooms in manager
```

Normally, in the event of OOM triggered by kube, we should see kubelet recording some signal for oom e.g. `An OOM event was triggered`

`Takeaway 3`: As far as Kube is concerned, the pod is well behaved and its all hakuna matata!
> ![](https://i.pinimg.com/originals/7a/18/f9/7a18f9c6efe7a954a42473cf8a5bd1fb.gif)

So, #Whodunit? Enter day 3 - new day new investigation 

## App's on Kube: day 3
Based on previous 3 takeaways, the only potential suspect we have is OS kernel. The pods are still crashing and metrics, events and kube level logs
do not justify the observation. 
  
### Reading kernel logs
1. System level log scan `grep -i -r 'out of memory' /var/log/` takes us somewhere. 
    ```
    /var/log/kern.log:Mar  9 13:17:05 ip-172-xxx-xx-xxx kernel: [30320.358563] Memory cgroup out of memory: Kill process 11190 (app) score 9 or sacrifice child
    ```
    `Takeaway 4`: We do in fact have kernel thinking memory cgroup is in danger and starting to kill!

2. Kernel logs (`/var/log/kern.log`) seem to have much more insightful info than above one liner than `out of memory: Kill process `. 
    
But before we look into this, lets do a bit of a deep dive into related concepts: 

### Deep-dive into OS Kernel

1. **Swap space and Kube**

   Docker supports setting [swappiness] however its discouraged as its slow and less performant. 
   Also, providing limit on swap is unsupported which can lead to resource management and overcommitment chaos. 
   These are one of the reasons why [kops] and in general Kube prefer no swap on hosts. 

2. **OOMKill disable on Kubernetes**

   OS Kernels allow disabling OOM Kill for selected cgroups level (`/sys/fs/cgroup/memory/memory.oom_control`) even docker 
   supports it using `--oom-kill-disable` flag. These are highly discouraged due to the nature of problem band-aid fixer `OOM Killer`
   solves. Its also, does not sit with Kube's declarative approach orchestration and also with cattle workload philosophy.
   Its also why, by default oom kill is enabled on kubernetes.
   
   Its possible however to configure it to disable OOMKill by starting kubelet service with `--cgroup-driver=cgroupfs` argument and ten setting `oom_kill_disable` under `/sys/fs/cgroup/memory/memory.oom_control`  as 1.
   
   `Takeaway 5`: Its not something I want to enable either, but completeness of the discussion, its worth mentioning :).    

3. **Kernel memory management**
    
    Kernel uses virtual addressing (using paging and segmentation) to provide isolation amongst various processes running on host. 
    It is also virtual addressing that allows for use of more memory than whats available currently in physical memory (RAM) by making use of other sources like disk (a.k.a. swap). 
    Virtual addressing is divided into user & kernel space. 
    User space is the sort of virtual address space that's reserved for user/application program whereas kernel space is reserved for kernel related operations. 
    
    Now, the os kernel is designed to be greedy - greedy to be able to run as many processes as possible. This is also the reason why we need mechanism like `out of memory'.
    
    
4. **System vs memory controller (memch) OOM**

   cgroups comprises of two components: `core` and `controller`. Core corresponds to managing the hierarchy and core capabilities whereas controllers
    are focused on type of resource cgroup is controlling eg cpu, io, memory controller ('memcg').  
    
    Now, the user-space out-of-memory handling can address OOM conditions for both cgroups using the memory controller ('memcg') and for the system as a whole.
    `Takeaway 6`: We know, based on our takeaways, that our OOM is not stemming from system draining or system as whole. Also, log `Memory cgroup out of memory` indicating that its `memcg`
    that's triggering the OOM Kill. Here, the app process hierarchy memory usage is aggregated together into its memcgs so the memory usage at group level can be accounted for. 
    What our first log here is telling us is `memcg usage reached its limits and memory cannot be reclaimed i.e. the memcg is out of memory`<sup>[1][lwn]</sup>.
    

6. **OOM kill score**

    How does kernel come to decide which process to kill, is based on a score. The score has two parts: main (`oom_score`) and adjustment factor (`oom_score_adj`).
    These scores are store against process id in process space and can be located on disk as :
    ```bash
    /proc/<pid>/oom_score
    /proc/<pid>/oom_score_adj
    ```
    
    The `oom_score` is given by kernel and is proportional to the amount of memory used by the process i.e. = 10 x percentage of memory used by process. 
    This means, the maximum `oom_score` is 100% x 10 = 1000!.
    Now, higher the oom_score higher the change of the process being killed. However, user can provide an adjustment factor `oom_score_adj (a.k.a. oom_adj in older kernel versions)`. 
    If provided, it is used to adjust the final score. The valid value for `oom_score_adj` is in range of  (-1000, +1000), where -ve score decreases and +ve increases the chances of oomkill.
    More details on this can be found in this very interesting article by Jonathan Corbet [another OOMKill rewrite][another oomkill rewrite], with precursory article found [here][oomkill writeup].

    
4. **OOM trigger workflow**

   `kmsg` is the kernel message interface that direct kernel messages to `/proc/kmsg` & `/dev/kmsg`. Now, `/dev/kmsg` is more useful as its designed to be persistent.
   `/proc/kmsg` is designed to be read once and treated more as event queue if you will. Messages from here also trickles through to kernel logs @ `/var/log/kern.log`.
   
    _On Kube_
    
    Kebelet watch for `kmsg` and handle messages that that will translate to OOMEvent/OOMKillEvent in Kube eventstream which is then handled appropriately 
    to trigger OOMKill. More interesting detail of how this happens can be found [here][line-eng-qos] (also shown in borrowed fig 9). 

    > ![](/images/oom/workflow-4-1024x816.png)
    *Figure 9: OOM handling workflow on Kubernetes. Image credit: [Line Corp][line-eng-qos]*

    As mentioned in `takeway 3 & 4`, this workflow however was not triggered in our case, we are did not record any Kube related OOM events or even kubelet receiving
    any related messages.
    
    _At Kernel Level_
    When system oor memory controller related OOM is suspected, based on `oom_score` (with adjustment `oom_score_adj`), `oom-killer` is invoked on highest
    score process and its children. 
    
    
    
In my case, memory cgroup ran out of memory and my stack trace confirms this (see fig 10). It tells me application container was 
killed because it was consuming 1.5MB shy of memory set as limit (31457280 KB).

> ![](/images/oom/log-part-1.jpg)
*Figure 10: Kernel log part 1*

OK! this explains the OOMKill but why:
a. My monitoring only show 29GB as max memory surge!
b. I never noticed beyond 9GB usage in local/testing/profiling and all the jazz!

> ![](/images/oom/2efa70f25d30b6e591150bc7a03e76e9-sticker.jpg)

This simply does not add up! Lets hold on to this thought for a bit and look at rest of the logs and what it says:

Before we go into part 2 of log, I should explain a few things:

1. The _pause container_ It is the parent container of each pod, responsible for creating and managing the environment for the group of containers 
that would be provisioned within the pod. For more info, I will direct you to an excellent article by [Ian Lewis], the [almighty pause container]. 
I need to explain this because it will be shown in following log.

2. Definition of memory cgroups stats metrics as per [kernel.org] is listed below. 
   
   Note that, `anonymous memory` (abbreviated often as `anon`) is a memory mapping with no file or device backing it. 
   Anon memory is used by programs to allocate memory for the stacks and heaps.
   Also, standard page size on linux kernel is 4KB which can be really inefficient to store mapping for large block of memory virtual memory. 
   `Hugepages` are designed to solve this inefficiency and can hold bigger chunk than 4KB. More details on this is available [here][hugepages]. 
   
    | Metrics of memory cgroups stats        | Definition           |
    | ------------- |:-------------:|
    |rss		| rss stands for resident set size. It is the portion of memory occupied by a process that is held in RAM. This metric represents number of bytes of anonymous and swap cache memory (includes transparent hugepages).|
    |rss_huge	| number of bytes of anonymous transparent hugepages.|
    |cache		| number of bytes of page cache memory.|
    |mapped_file	| number of bytes of mapped file (includes tmpfs/shmem)|
    |swap		| number of bytes of swap usage|
    |dirty		| number of bytes that are waiting to get written back to the disk.|
    |writeback	| number of bytes of file/anon cache that are queued for syncing to disk.|
    |inactive_anon	| number of bytes of anonymous and swap cache memory on inactive LRU list.|
    |active_anon	| number of bytes of anonymous and swap cache memory on active LRU list.|
    |inactive_file	| number of bytes of file-backed memory on inactive LRU list.|
    |active_file	| number of bytes of file-backed memory on active LRU list.|
    |unevictable	| number of bytes of memory that cannot be reclaimed (mlocked etc).|


Now, as discussed previously, swap is not being used in this system. See the second part of the logs in fig 11. 
You will note, there are two containers recorded and their memory stats is capture - a) the pause container and b) the app container. 
We can ignore pause, its tiny and looking very healthy. But look at the stats for app pod in fig 11 (below)!
At the time my app was killed, it held about 29GB in hugepages and only 1.3GB extra in RSS. 
That's huge and remember monitoring it not picking it for some reason! It captured 29GB but not 31GB! Perhaps its picking only `rss_huge` and
presenting it as `rss` erroneously! `¯\_(ツ)_/¯`! Yes we have a problem but this monitoring issue is for another day!

> ![](/images/oom/log-part-2.jpg)
*Figure 11: Kernel log part 2*

Notice the blue arrow in fig 11, its capturing page info by both the pause container process and app container process. These are page info and not
and need to be multiplied by 4KB to get actual memory stats. They are translated two lines below the blue line! 

My app has freaking **_62GB_** in total virtual memory! Whats going on!
> ![](/images/oom/wth.gif)

Ok, so "total-vm" is the part of virtual memory the process uses. A part of this "total-vm" that's mapped to RAM is `rss`. Part of `rss` that's allocated on
to real memory blocks is your `anon-rss` (anonymous memory) and the other part of rss is mapped to devices and files and termed `file-rss`.
If my app goes crazy and allocates large chunk of space (say using malloc()) but never really use it then `total-vm` can be high but it wont all be used in real memory. 
This is made possible due to overcommit. A good sign of this happening, given swap off, is when `total-vm` is high but `rss` is actually low!
This is exactly whats happening here! We have about **30GB** difference between `total-vm` and `rss`.

`Takeaway 7`: We have two problems here: a) Supporting over-commitment and b) Allocation of what we suspect un-needed memory!   

Let's look at solving the over-commit first and see what level of fixes it provides:
 
### Controlling over-commits

So far, we concluded over-commitment as a problem. In fact, as discussed previously, its a feature (of both kernel & kube)! 

> ![](https://memegenerator.net/img/instances/80565312/its-not-a-bug-its-a-feature.jpg)

Kernel uses the "extendability" of virtual addressing to over-commit. The kernel setting `vm.overcommit_memory` and 
`vm.overcommit_ratio` are specially designed to controlling this capability. For more info, see [here][problem with overcommit].

1.1 `vm.overcommit_memory = 0`: Make best guess and overcommit where possible. This is default.

1.2 `vm.overcommit_memory = 1`: Always overcommit 

1.3 `vm.overcommit_memory = 2`: Never overcommit, and only allocate as much memory as defined in overcommit_ratio.

`vm.overcommit_ratio` is only used when overcommit_memory=2. It defines what percent of the physical RAM plus swap space should be allocated. This is default to 50.
We want this config to be 100. 

But use of `sysctl` to set these(using following) is not enough as the config wont persist on horizontal scaling (new node spinning due to spot instances or less important but restart):
```bash
sysctl -w vm.overcommit_memory=2
sysctl -w vm.overcommit_ratio=100
```
The effect of these config is immediate and not start is needed. Talking about restart, `systcl` cli config update do not persist to persist, 
system config need to be updated `/etc/sysctl.conf` to persist the setting across restarts. 

On `Kube`, [kops] provisioned clusters, these settings needs to be supplied through [sysctlparameters] config but these 
are only supported from kube 1.17 and higher! Safe [sysctl parameters can be set at pod level][sysctl config on pods] however
our setting is not (obviously) supported at pod level.

and this cluster is currently at 1.12! Heya, Mr Murphy!

> ![](/images/oom/mrmurphy.jpeg)

So, I say our my prayers, and turn to bash:
```bash
for memip in $(aws ec2 describe-instances --region us-east-1 --instance-ids \
$(aws autoscaling describe-auto-scaling-instances --region us-east-1 --output text \
--query "AutoScalingInstances[?AutoScalingGroupName=='myasg'].InstanceId") \
--query "Reservations[].Instances[].PrivateIpAddress")
do 	
    ssh -o StrictHostKeyChecking=no  ${memip} 'bash -s' < set_mem.sh	
done 
``` 
where `set_mem.sh` is:
```bash
#!/usr/bin/env bash
sudo sysctl -w vm.overcommit_memory=2
sudo sysctl -w vm.overcommit_ratio=100 
```

I see massive improvement in OOMKills. Pods that were killed every 20mins and odd, are running 24h
> ![](/images/oom/app-no-crash.jpg)
*Figure 12: Getting somewhere! OOMKills sort of under control!*

I am not done yet however! Remember, part `b` of our problem in `takeaway 7` i.e. `b) Allocation of what we suspect un-needed memory!`.

Why was it happening in the first place, and why its controlled with overcommit disabled (remember itsnot cured at root, still happens far less however)!
Oh the fun never ends! All the places we go! 
I will cover this later, ahem ahem, when I know the answer! Pretty sure its some nasty behaviour of Tensorflow 2, and investigation is _underway_!

> ![](/images/oom/4882580.jpg)

Thanks for reading. Hopefully it was fun insightful read!
 
[1]: //suneeta-mall.github.io/talks/KubernetesSydneyForum_AU_2019.html
[2]: //suneeta-mall.github.io/talks/KubeCon_US_2019.html
[numpy]: //numpy.org
[Tensorflow]: //tensorflow.org
[container runtime]: //www.ianlewis.org/en/container-runtimes-part-4-kubernetes-container-run
[Ian Lewis]: //twitter.com/IanMLewis
[container from scratch]: //www.youtube.com/watch?v=8fi7uSYlOdc
[Liz Rice]: //twitter.com/lizrice
[source]: //github.com/lizrice/containers-from-scratch/blob/master/main.go
[sysdig's]: https://sysdig.com/blog/troubleshoot-kubernetes-oom/
[line-eng-qos]: https://engineering.linecorp.com/en/blog/prometheus-container-kubernetes-cluster/
[Container Advisor]: //github.com/google/cadvisor
[cgroups]: https://man7.org/linux/man-pages/man7/cgroups.7.html
[requests and limits]: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
[best practices resource requests and limits]: https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-resource-requests-and-limits
[problem with overcommit]: https://engineering.pivotal.io/post/virtual_memory_settings_in_linux_-_the_problem_with_overcommit/
[lwn]: https://lwn.net/Articles/590960/
[another oomkill rewrite]: https://lwn.net/Articles/391222/
[oomkill writeup]: https://lwn.net/Articles/317814/
[Docker: Up & Running]: https://www.oreilly.com/library/view/docker-up/9781492036722/
[swappiness]: https://docs.docker.com/config/containers/resource_constraints/
[kops]: https://github.com/kubernetes/kops/issues/3251
[sysctl config on pods]: https://kubernetes.io/docs/tasks/administer-cluster/sysctl-cluster/
[sysctlparameters]: https://github.com/kubernetes/kops/blob/master/docs/cluster_spec.md#sysctlparameters
[almighty pause container]: https://www.ianlewis.org/en/almighty-pause-container
[kernel.org]: https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt
[hugepages]: https://docs.openshift.com/container-platform/4.1/scalability_and_performance/what-huge-pages-do-and-how-they-are-consumed-by-apps.html