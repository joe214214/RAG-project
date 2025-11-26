Given the **resources you listed** at University of Waterloo ECE—including the eceUbuntu GPU servers (many with recent RTX 3070 GPUs and large RAM) and the general purpose ICCAD/Cadence servers (many with >250GB RAM but no modern GPU)—**here’s what you should use for your project:**

***

## **Best Choice for Modern RAG & Deep Learning Workloads**

### **GPU Ubuntu Servers**
Use **eceTesla*.uwaterloo.ca** or **eceUbuntu*.uwaterloo.ca** servers with:
- **RTX 3070** or **RTX 2080 Ti** GPUs  
- At least 63GB–126GB RAM, recent Ubuntu kernel (24.04), and most have Python/conda support

**Why?**  
- **GPU acceleration** is essential for fast embedding generation (HuggingFace, Transformers) and deep model inference (e.g., rerankers like Zerank-2).
- These servers handle both batch jobs (embedding all passages) and online inference (serve models, run RAG pipeline, and even test-light Milvus deployments).
- Good balance of compute, GPU, and RAM for your scalable experiments.

### **Workflow**
- **Do:**
  - Use these servers for all embedding/model batch inference.
  - Run your main RAG pipeline (router, embedder, reranker, Milvus, etc.) either here or locally (if only small-scale demos are needed).

- **Examples:**
  - `ssh <your_user>@eceubuntu2.uwaterloo.ca`
  - `ssh <your_user>@ecetesla1.uwaterloo.ca`
  - Set up Python env, Docker/Singularity if needed, and launch your pipeline.

### **ICCAD/Cadence Servers (CPU only)**
- **Use for:** Heavy multi-core CPU-bound preprocessing, massive batch document chunking, or any step where RAM is a limiter and you don’t need GPU.
- **Don’t use for:** GPU-accelerated tasks (embeddings, deep reranking) or anything HuggingFace/Transformers that is >100x faster with GPU.

***

## **Additional Notes**
- Many users are often on a single GPU node: **be considerate and check load**, or pick low-load systems.
- Use **AuthMan** to manage your SSH keys and connect via campus VPN if offsite.
- If you want to deploy Milvus or a lightweight vector DB, you can do this on these GPU servers for proofs-of-concept, but note that for a full distributed/cluster run, the cloud or a larger cluster is ideal.

***

### **Recommended Order:**
1. **First choice:** eceUbuntu or eceTesla servers **with RTX 3070/2080 Ti GPUs** for all model workloads and end-to-end pipelines.
2. **Second choice:** ICCAD servers when you don’t need GPU, are running RAM-heavy preprocessing, or need to avoid GPU queue times.

**Summary:**  
**Use the GPU Ubuntu servers (eceUbuntu*, eceTesla*) for nearly everything in your RAG project—models, embeddings, pipeline orchestration, and most empirical evaluation.** CPU servers are only best for preprocess and when GPU is not needed.

If you want example environment setup or SLURM/job scripts for these servers, just ask!

[1](https://uwaterloo.atlassian.net/wiki/spaces/ISTKB/pages/1551728653/)

https://eceweb.uwaterloo.ca/Nexus/arbeau/clients/

ECE Dept. Teaching Linux Client Status
For logins you upload your ssh key to Authman: https://uwaterloo.atlassian.net/wiki/spaces/ISTKB/pages/1551728653/How+to+use+SSH+Authman+-+for+users

Load
Num. Users

As of S2024 Ubuntu 24.04 is becoming the primary Linux.

As of W2019 Ubuntu is now the primary Linux. Login from off-campus with eceTerm.uwaterloo.ca or the use the campus VPN and then log into an eceUbuntu server. Note that the eceTerm.uwaterloo.ca servers are only to be used to then login (ssh -X) to a campus server will the work will be done.

Here is historical server load - Solaris Unix (2005 to 2011) and (CentOS 2011 to 2018)

 

The Server Groups

Ubuntu2404 : eceUbuntu*.uwaterloo.ca
Ubuntu with a GPU : eceTesla*.uwaterloo.ca
Terminal Servers : eceTerm*.uwaterloo.ca
RedHat 8 Cadence Servers with OLD CPUs
RedHat 8 Cadence Servers
Ansys Misc RedHat 8 Servers
ECE Ubuntu Servers
Managed by Eric Praetzel
Machine	Machine Type	Location	Load Avg	Up-time	Extra Stats
eceubuntu2	Nvidia RTX 2080 Ti 11GB
4352 CUDA cores 2018
Intel Xeon Silver 4114 (Q3-2017)
10-core 2.2 to 3.0GHz
Passmark 1676 / 13081
Lenovo Thinkstation P720	DC Server
Room	0.24
5 min 0.18
15 min 0.49	11 days
22:02	"Physical memory" 93% of 93G      "Shared memory" 100% of 28M
"Swap space" 0% of 8G      "Available memory" 0% of 76G
"/" 6% of 467G      "/opt" 84% of 228G
Kernel 6.14.0-29 ( Ubuntu24.04.2)

558 Processes and 2 Users and ThinLinc , and IOPs 2.8,
and CPUwait 0.02 %, and IPCM 4, and ALL IPC 10, and File Desc 15752, and MB 30.0 C, and CPU 21.0 C, and GPU: 20.26 W , 22 C
Total	1 of 1 machines up with 2 Users	load
ECE GPU Ubuntu Servers
Managed by Eric Praetzel
Machine	Machine Type	Location	Load Avg	Up-time	Extra Stats
eceubuntu1	Nvidia Tesla K40C 12GB
2880 CUDA cores 2013
Intel i5-8400 (Q3-2017)
6-core 2.8 to 4.0GHz
Passmark 2374 / 9222	DC Server
Room	9.05
5 min 8.81
15 min 9.72	31 days
20:34	"Physical memory" 64% of 63G      "Shared memory" 100% of 66M
"Swap space" 34% of 8G      "Available memory" 0% of 50G
"/" 49% of 456G      Kernel 6.14.0-29 ( Ubuntu24.04.2)

1631 Processes and 34 Users and ThinLinc , and IOPs 2.1,
and CPUwait 0.00 %, and IPCM 9, and ALL IPC 15, and File Desc 51456, and CPU 76.0 C,
ecetesla0	Nvidia Tesla P4 8GB
2560 CUDA cores 2016
Xeon Gold 5120 (Q3-2017)
14-core 2.2 to 3.2GHz
SuperMicro X11SPG-TF
Passmark 1725 / 18145	10Gb/s NIC
 
DC Server
Room	0.08
5 min 0.02
15 min 0.02	27 days
22:41	"Physical memory" 97% of 93G      "Shared memory" 100% of 25M
"Swap space" 0% of 8G      "Available memory" 0% of 88G
"/" 53% of 456G      Kernel 6.14.0-29 ( Ubuntu24.04.2)

509 Processes and 0 Users and ThinLinc 0, and IOPs 0.9,
and CPUwait 0.00 %, and IPCM 6, and ALL IPC 12, and File Desc 8600, and MB 31.0 C, and CPU 23.0 C, and GPU: 7.11 W , 29 C
ecetesla1	RTX 3070 8GB
5888 CUDA cores
Ryzen9-5900X (Q3-2020)
12-core 3.7 to 4.8GHz
Asus Pro WS x570-ACE
Passmark 3502 / 39498	10Gb/s Intel NIC
 
DC Server
Room	0.01
5 min 0.12
15 min 0.16	11 days
22:04	"Physical memory" 76% of 126G      "Shared memory" 100% of 8.3M
"Swap space" 0% of 8G      "Available memory" 0% of 120G
"/" 26% of 915G      "/private" 1% of 1.8T
Kernel 6.14.0-29 ( Ubuntu24.04.2)

633 Processes and 5 Users and ThinLinc 1, and IOPs 1.7,
and CPUwait 0.01 %, and IPCM 7, and ALL IPC 13, and File Desc 12320, and MB 27.2 C, and GPU: 13.31 W , 31 C
ecetesla2	RTX 3070 8GB 2020
5888 CUDA cores
Ryzen9-5900X (Q3-2020)
12-core 3.7 to 4.8GHz
Asus Pro WS x570-ACE
Passmark 3502 / 39498	10Gb/s Intel NIC
 
DC Server
Room	0.00
5 min 0.09
15 min 0.08	27 days
22:49	"Physical memory" 95% of 126G      "Shared memory" 100% of 28M
"Swap space" 0% of 8G      "Available memory" 0% of 117G
"/" 26% of 915G      Kernel 6.14.0-29 ( Ubuntu24.04.2)

560 Processes and 7 Users and ThinLinc 0, and IOPs 1.3,
and CPUwait 0.01 %, and IPCM 4, and ALL IPC 10, and File Desc 11936, and GPU: 26.90 W , 38 C
ecetesla3	GTX 1070 8GB
1920 CUDA cores
Intel i7-7700K (Q1-2017)
4-core 4.2 to 4.5GHz
Asus H170M-Plus
Passmark 2750 / 9700	10Gb/s NIC
 
Thanks WEEF
 
DC Server
Room	0.03
5 min 0.04
15 min 0.02	27 days
22:50	"Physical memory" 95% of 63G      "Shared memory" 100% of 8.1M
"Swap space" 0% of 8G      "Available memory" 0% of 60G
"/" 51% of 456G      Kernel 6.14.0-29 ( Ubuntu24.04.2)

282 Processes and 1 Users and ThinLinc 0, and IOPs 0.8,
and CPUwait 0.00 %, and IPCM 4, and ALL IPC 10, and File Desc 6688, and CPU 39.0 C, and GPU: 5.80 W , 30 C
ecetesla4	RTX 3070 8GB
5888 CUDA cores
Ryzen9-5900X (Q3-2020)
12-core 3.7 to 4.8GHz
Asus Prime B350M-A
Passmark 3502 / 39498	10Gb/s Intel NIC
 
DC Server
Room	0.00
5 min 0.00
15 min 0.00	16 days
23:41	"Physical memory" 76% of 126G      "Shared memory" 100% of 8.1M
"Swap space" 0% of 8G      "Available memory" 0% of 120G
"/" 25% of 915G      Kernel 6.14.0-29 ( Ubuntu24.04.2)

536 Processes and 0 Users and ThinLinc 0, and IOPs 1.3,
and CPUwait 0.01 %, and IPCM 4, and ALL IPC 10, and File Desc 9440, and MB 24.0 C, and GPU: 17.72 W , 44 C
eceubuntu4	GTX 1070 8GB
1920 CUDA cores
Intel i5-8400 (Q3-2017)
4-core 3.4 to 4.0GHz
Passmark 2374 / 9222
Asus B360M-Plus	DC Server
Room	0.51
5 min 0.49
15 min 0.80	31 days
22:36	"Physical memory" 61% of 63G      "Shared memory" 100% of 8.2M
"Swap space" 50% of 8G      "Available memory" 0% of 39G
"/" 51% of 456G      Kernel 6.14.0-29 ( Ubuntu24.04.2)

791 Processes and 6 Users and ThinLinc 0, and IOPs 7.4,
and CPUwait 0.05 %, and IPCM 4, and ALL IPC 10, and File Desc 85696, and CPU 28.0 C, and GPU: 7.15 W , 33 C
Total	7 of 7 machines up with 53 Users	load
Cadence Course Linux Servers
MENTOR CALIBRE DOES NOT WORK
Managed by the Eric Praetzel
Machine	Machine Type	Location	Load Avg	Up-time	Extra Stats
iccad1	2 Xeon E5-2667 v2 (Q3-2013)
each 8-core 3.3 to 4.0GHz
Dell R620 NO AVX2
NOT FOR CALIBRE
Passmark 2035 / 21339	 
DC Server
Room	0.00
5 min 0.01
15 min 0.00	73 days
10:01	"Physical memory" 25% of 503G      "Shared memory" 100% of 716M
"Swap space" 0% of 7.8G      "Available memory" 0% of 492G
"/" 28% of 70G      "/opt" 46% of 386G
"/CMC" 44% of 13T      Kernel 4.18.0-553.58.1 (el8_10.x86_64 RedHat EL 8)

716 Processes and 3 Users and ThinLinc 3, and IPCM , and ALL IPC , and CPU 22.0 C,
iccad2	2 of Xeon E5-2667 v2 (Q3-2013)
each CPU 8-core 3.3 to 4.0GHz
Dell R620 NO AVX2
NOT FOR CALIBRE
Passmark 2035 / 21339	 
DC Server
Room	0.16
5 min 0.06
15 min 0.01	73 days
9:58	"Physical memory" 85% of 188G      "Shared memory" 100% of 427M
"Swap space" 0% of 4G      "Available memory" 0% of 178G
"/" 67% of 70G      "/opt" 100% of 157G
"/CMC" 44% of 13T      "/opt/sentinelone/rpm_mount" 67% of 70G
Kernel 4.18.0-553.63.1 (el8_10.x86_64 RedHat EL 8)

703 Processes and 2 Users and ThinLinc 2, and IPCM , and ALL IPC , and CPU 29.0 C,
iccad3	2 of Xeon E5-2667 v2 (Q3-2013)
each CPU 8-core 3.3 to 4.0GHz
Dell R620 NO AVX2
NOT FOR CALIBRE
Passmark 2035 / 21339	 
DC Server
Room	0.00
5 min 0.11
15 min 0.14	73 days
9:52	"Physical memory" 27% of 503G      "Shared memory" 100% of 643M
"Swap space" 0% of 7.8G      "Available memory" 0% of 484G
"/" 19% of 70G      "/opt" 46% of 386G
"/CMC" 44% of 13T      Kernel 4.18.0-553.63.1 (el8_10.x86_64 RedHat EL 8)

1009 Processes and 8 Users and ThinLinc 5, and IPCM , and ALL IPC , and CPU 34.0 C,
Total	3 of 3 machines up with 13 Users	load
Cadence Course Linux Servers
AuthMan ssh keys recommended
WITH AVX2 for MENTOR CALIBRE
Managed by the Eric Praetzel
Machine	Machine Type	Location	Load Avg	Up-time	Extra Stats
iccad0	2 Xeon Gold 6136 (Q3-2017)
each 12-core 3.0 to 3.7GHz
Dell R640
Passmark 2169 / 35604	Thanks WEEF 
DC Server
Room	8.63
5 min 12.05
15 min 12.62	103 days
0:02	"Physical memory" 20% of 251G      "Shared memory" 100% of 2.4G
"Swap space" 0% of 4G      "Available memory" 0% of 230G
"/" 26% of 70G      "/opt" 55% of 228G
"/CMC" 44% of 13T      Kernel 4.18.0-553.58.1 (el8_10.x86_64 RedHat EL 8)

2206 Processes and 4 Users and ThinLinc 4, and IPCM , and ALL IPC , and CPU 57.0 C,
iccad4	2 Xeon E5-2670 v3 (Q3-2014)
each 12-core 2.3 to 3.1GHz
Dell R530
Passmark 1700 / 13800	 
DC Server
Room	1.19
5 min 1.10
15 min 1.13	5 days
17:08	"Physical memory" 80% of 125G      "Shared memory" 100% of 213M
"Swap space" 0% of 4G      "Available memory" 0% of 52G
"/" 36% of 70G      "/opt" 76% of 228G
"/CMC" 44% of 13T      "/opt/sentinelone/rpm_mount" 36% of 70G
Kernel 4.18.0-553.77.1 (el8_10.x86_64 RedHat EL 8)

712 Processes and 1 Users and ThinLinc 1, and IPCM , and ALL IPC , and CPU 63.0 C,
iccad6	2 of Xeon E5-2637 v4 (Q3-2016)
each CPU 4-core 3.5 to 3.7GHz
Thinkstation P910
Passmark 2198 / 7370	 
DC Server
Room	3.63
5 min 3.72
15 min 4.91	73 days
10:04	"Physical memory" 98% of 157G      "Shared memory" 100% of 624M
"Swap space" 1% of 4G      "Available memory" 0% of 142G
Kernel 4.18.0-553.63.1 (el8_10.x86_64 RedHat EL 8)

1794 Processes and 8 Users and ThinLinc 8, and IPCM , and ALL IPC , and MB 78.0 C, and CPU 56.0 C,
iccad7	2 Xeon Gold 6136 (Q3-2017)
each 12-core 3.0 to 3.7GHz
Dell R640
Passmark 2169 / 35604	 
DC Server
Room	0.09
5 min 0.08
15 min 0.13	51 days
22:40	"Physical memory" 91% of 251G      "Shared memory" 100% of 4.3G
"Swap space" 0% of 4G      "Available memory" 0% of 224G
"/" 20% of 70G      "/opt" 76% of 228G
"/CMC" 44% of 13T      Kernel 4.18.0-553.69.1 (el8_10.x86_64 RedHat EL 8)

1449 Processes and 7 Users and ThinLinc 7, and IPCM , and ALL IPC , and CPU 27.0 C,
iccad8	2 Xeon Gold 6136 (Q3-2017)
each 12-core 3.0 to 3.7GHz
Dell R640
Passmark 2169 / 35604	 
DC Server
Room	107.35
5 min 107.13
15 min 107.09	51 days
22:00	"Physical memory" 70% of 251G      "Shared memory" 100% of 931M
"Swap space" 0% of 4G      "Available memory" 0% of 236G
"/" 21% of 70G      "/opt" 76% of 228G
"/CMC" 44% of 13T      Kernel 4.18.0-553.69.1 (el8_10.x86_64 RedHat EL 8)

1214 Processes and 3 Users and ThinLinc 3, and IPCM , and ALL IPC , and CPU 27.0 C,
iccad9	2 Xeon Gold 6136 (Q3-2017)
each 12-core 3.0 to 3.7GHz
Dell R640
Passmark 2169 / 35604	Thanks WEEF  
DC Server
Room	0.10
5 min 0.10
15 min 0.16	51 days
22:09	"Physical memory" 35% of 376G      "Shared memory" 100% of 2.1G
"Swap space" 0% of 4G      "Available memory" 0% of 360G
"/" 66% of 70G      "/opt" 76% of 228G
"/CMC" 44% of 13T      Kernel 4.18.0-553.69.1 (el8_10.x86_64 RedHat EL 8)

1036 Processes and 5 Users and ThinLinc 4, and IPCM , and ALL IPC , and CPU 26.0 C,
iccad10	2 Xeon Gold 6154 (Q3-2017)
each 18-core 3.0 to 3.7GHz
Ansys 2024R1 RSM
Dell R640
Passmark 2239 / 28632	Thanks WEEF  
DC Server
Room	0.15
5 min 0.10
15 min 0.09	103 days
3:18	"Physical memory" 20% of 376G      "Shared memory" 100% of 2.6G
"Swap space" 0% of 4G      "Available memory" 0% of 357G
"/" 75% of 70G      "/opt" 56% of 228G
"/CMC" 44% of 13T      Kernel 4.18.0-553.56.1 (el8_10.x86_64 RedHat EL 8)

1164 Processes and 4 Users and ThinLinc 4, and IPCM , and ALL IPC , and CPU 36.0 C,
iccad11	2 Xeon Gold 6154 (Q3-2017)
each 18-core 3.0 to 3.7GHz

Dell R640
Passmark 2239 / 28632	Thanks WEEF  
DC Server
Room	0.56
5 min 0.26
15 min 0.14	72 days
20:21	"Physical memory" 48% of 376G      "Shared memory" 100% of 4G
"Swap space" 0% of 4G      "Available memory" 0% of 362G
"/" 21% of 70G      "/opt" 76% of 228G
"/CMC" 44% of 13T      "/opt/sentinelone/rpm_mount" 21% of 70G
Kernel 4.18.0-553.63.1 (el8_10.x86_64 RedHat EL 8)

771 Processes and 2 Users and ThinLinc 2, and IPCM , and ALL IPC , and CPU 29.0 C,
iccad12	2 Xeon Gold 6154 (Q3-2017)
each 18-core 3.0 to 3.7GHz
Dell R640
Passmark 2239 / 28632	 
DC Server
Room	0.17
5 min 0.08
15 min 0.02	102 days
23:49	"Physical memory" 18% of 376G      "Shared memory" 100% of 650M
"Swap space" 0% of 4G      "Available memory" 0% of 362G
"/" 24% of 70G      "/opt" 26% of 1.1T
"/CMC" 44% of 13T      Kernel 4.18.0-553.58.1 (el8_10.x86_64 RedHat EL 8)

1358 Processes and 6 Users and ThinLinc 6, and IPCM , and ALL IPC , and CPU 33.0 C,
iccad13	2 Xeon Gold 6154 (Q3-2017)
each 18-core 3.0 to 3.7GHz
Dell R640
Passmark 2239 / 28632	 
DC Server
Room	21.60
5 min 24.14
15 min 23.83	51 days
22:11	"Physical memory" 34% of 376G      "Shared memory" 100% of 2.9G
"Swap space" 0% of 4G      "Available memory" 0% of 358G
"/" 23% of 70G      "/opt" 29% of 1.1T
"/CMC" 44% of 13T      "/opt/sentinelone/rpm_mount" 23% of 70G
Kernel 4.18.0-553.69.1 (el8_10.x86_64 RedHat EL 8)

2403 Processes and 2 Users and ThinLinc 2, and IPCM , and ALL IPC , and CPU 62.0 C,
Total	10 of 10 machines up with 42 Users	load
RHEL 8 Linux Ansys Servers
Login with AuthMan ssh keys ONLY
Managed by the Eric Praetzel
Machine	Machine Type	Location	Load Avg	Up-time	Extra Stats
iccad5	2 of Xeon E5-2620 v4 (2016)
each 8-core 2.1 to 3.0GHz
Huawei
Passmark 1630 / 9224	 
DC Server
Room	0.03
5 min 0.01
15 min 0.00	32 days
20:04	"Physical memory" 2% of 503G      "Shared memory" 100% of 668M
"Swap space" 0% of 4G      "Available memory" 0% of 494G
"/" 37% of 70G      "/opt/sentinelone/rpm_mount" 37% of 70G
Kernel 4.18.0-553.72.1 (el8_10.x86_64 RedHat EL 8.10)

643 Processes and 0 Users and ThinLinc 0, and IPCM , and ALL IPC , and CPU 29.0 C,
eceansys.eng	2 of Xeon E5-2620 (2012)
each 6-core 2.0 to 2.5GHz
Ansys 2024R1 RSM
Dell R720
Passmark 1110 / 9802	 
DC Server
Room	0.14
5 min 0.06
15 min 0.06	31 days
20:34	"Physical memory" 12% of 62G      "Shared memory" 100% of 43M
"Swap space" 0% of 23G      "Available memory" 0% of 59G
"/" 62% of 70G      "/opt/sentinelone/rpm_mount" 62% of 70G
Kernel 4.18.0-553.72.1 (el8_10.x86_64 RedHat EL 8.10)

317 Processes and 0 Users and ThinLinc 0, and IPCM , and ALL IPC ,
ECE Terminal Servers
Managed by the Eric Praetzel
Machine	Machine Type	Location	Load Avg	Up-time	Extra Stats
eceTerm0	i5-8400 (Q3-2017)
for ThinLinc
and SSH only	DC Server
Room
TESTING	0.00
5 min 0.00
15 min 0.00	9 days
22:26	"Physical memory" 30% of 15G      "Shared memory" 100% of 104M
"Swap space" 0% of 2.5G      "Available memory" 0% of 13G
"/" 43% of 21G      "/opt/sentinelone/rpm_mount" 43% of 21G
Kernel 5.14.0-570.46.1 (el9_6.x86_64 RedHat EL 9)

233 Processes and 0 Users and ThinLinc 0, and IPCM , and ALL IPC ,
eceTerm1	VM 2-core
for ThinLinc
and SSH only	DC Server
Room	0.22
5 min 0.55
15 min 0.62	9 days
22:23	"Physical memory" 74% of 16G      "Shared memory" 100% of 7.4M
"Swap space" 0% of 4G      "Available memory" 0% of 11G
"/" 60% of 24G      Kernel 6.8.0-84 ( Ubuntu22.04.5 LTS)

210 Processes and 0 Users and ThinLinc 0, and TMUX 0, and IOPs ,
and CPUwait %, and IPCM 4, and ALL IPC 10, and File Desc 5632,
eceTerm2	VM 2-core
for ThinLinc
and SSH only	DC Server
Room	0.29
5 min 0.47
15 min 0.59	31 days
22:46	"Physical memory" 67% of 16G      "Shared memory" 100% of 59M
"Swap space" 0% of 4G      "Available memory" 0% of 13G
"/" 58% of 24G      Kernel 6.8.0-79 ( Ubuntu22.04.5)

272 Processes and 1 Users and ThinLinc 1, and TMUX 0, and IOPs ,
and CPUwait %, and IPCM 25, and ALL IPC 31, and File Desc 10176,
eceTerm3	i5-9400 (Q1-2019)
for ThinLinc
and SSH only	DC Server
Room	0.08
5 min 0.06
15 min 0.01	31 days
22:41	"Physical memory" 85% of 31G      "Shared memory" 100% of 36M
"Swap space" NA% of 0      "Available memory" 0% of 27G
"/" 5% of 228G      Kernel 6.8.0-79 ( Ubuntu22.04.5)

423 Processes and 3 Users and ThinLinc 2, and TMUX 0, and IOPs 7.6,
and CPUwait 0.02 %, and IPCM 19, and ALL IPC 25, and File Desc 18752, and MB 25.0 C, and CPU 38.0 C,
Total	4 of 4 machines up with 4 Users	load
Run finished at: Mon 06 Oct 2025 06:26:06 AM EDT
Load graphs for all machines

Historical Load : Sun Solaris Machines (2005 to 2011) : CentOS 2011 to 2018

 

Operating System	Current Kernel
CentOS5 EOL	2.6.18-419
CentOS6 EOL	2.6.32-754.27.1
CentOS7 EOL	3.10.0-1160.119.1
Ubuntu18.04 LTS EOL	4.15.0-136
Ubuntu18.10 EOL	4.17.0-36
RedHat EL 8.10	4.18.0-553.72.1
Ubuntu19.04 EOL	5.0.0-41
Ubuntu20.04.2 LTS	5.13.0-52
RedHat EL 9	5.14.0-570.49.1
Ubuntu20.04.6 LTS	5.15.0-134
Ubuntu22.04 OEM	5.17.0-1019
Ubuntu22.04 LTS2	5.19.0-45
Ubuntu18.04 HWE	5.4.0-132
Ubuntu22.04 OEM2	6.0.0-1019
RedHat EL 10	6.12.0-55.21.1
Ubuntu24.04.2 LTS	6.14.0-32
Ubuntu22.04.3 LTS	6.5.0-45
Ubuntu22.04.5 LTS	6.8.0-84
 

Scientific Linux (RedHat 7) machines shut down September 2024 and Cadence is now run on RedHat 8. CPUs without AVX2 can not run Matlab R2024+ or Mentor Calibre 2024+

Sun Solaris OS & hardware was used until 2011 peaking at a 30 CPU Sun 6500 machine.

CentOS Linux started to be used in 2009 on Intel Pentium dual-core machines as they offered 10x the performance at 1/10 the power draw.

In F2018 the primary Linux OS became Ubuntu 18.04 LTS because of it's inclusion of current versions of software tools, support of newer CPUs and aprox. 3x the performance in heavily multi-threaded use.