# ECE Hadoop Cluster - Complete Guide

## 📝 About This Guide

This guide uses the following example credentials throughout all commands and examples:
- **UW User ID**: `oankit`
- **Local Machine**: Windows (`C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment`)
- **Shell**: Default cluster shell is `csh/tcsh` (not bash)

**⚠️ Important:** Replace `oankit` with your own UW user ID in all commands!

---

## Table of Contents
1. [Cluster Overview](#cluster-overview)
2. [Access Requirements](#access-requirements)
3. [Connecting to the Cluster](#connecting-to-the-cluster)
4. [Cluster Architecture](#cluster-architecture)
5. [Setting Up Your Environment](#setting-up-your-environment)
6. [Running MPI Jobs](#running-mpi-jobs)
7. [Transferring Files](#transferring-files)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Cluster Overview

The **ECE Hadoop Cluster** is a distributed computing resource provided by the University of Waterloo's Electrical and Computer Engineering (ECE) department for students and researchers.

### Cluster Specifications

**Total Resources:**
- **16 worker nodes** (ecehadoop0 through ecehadoop15)
- **64 total CPU cores** (4 cores per node)
- **~256 slots** for MPI jobs (4 slots per node)

**Individual Node Specs:**
- **Hostname Pattern**: `ecehadoopN.private.uwaterloo.ca` (N = 0 to 15)
- **CPU Cores**: 4 per node
- **MPI Slots**: 4 per node (max-slots=4)
- **Operating System**: Linux (likely CentOS/RHEL based)
- **Network**: Private network within UW infrastructure
- **Shell**: Default is `csh/tcsh` (not bash!)

**Software Available:**
- MPI implementations (OpenMPI)
- Python 3
- Standard development tools (gcc, make, etc.)
- pip for Python package management

---

## Access Requirements

### 1. UW Network Access

You must be either:
- **On-campus**: Directly connected to UW network
- **Off-campus**: Connected via UW VPN or using a jump host

### 2. ECE Account

- You should already have access via your UW user ID
- Professor confirmation: "You should have access to ecehadoop already"
- No special application required for course students
- **Example user ID used in this guide**: `oankit` (replace with your own UW user ID)

### 3. SSH Key Setup (Recommended)

The cluster uses **SSH Authman** for key-based authentication:
- Upload your SSH public key to UW's Authman system
- Reference: https://uwaterloo.atlassian.net/wiki/spaces/ISTKB/pages/1551728653/How+to+use+SSH+Authman+-+for+users

---

## Connecting to the Cluster

### Connection Methods

**Note:** Replace `YOUR_USERID` with your UW user ID in all examples below.
- **Example used throughout this guide**: `oankit`

#### Method 1: Direct (On-Campus Only)

```bash
# Generic format
ssh YOUR_USERID@ecehadoop.private.uwaterloo.ca

# Example with oankit
ssh oankit@ecehadoop.private.uwaterloo.ca
```

**Note:** This connects you to the master node.

#### Method 2: Via Jump Host (Off-Campus)

**Step 1:** Connect to ECE terminal server first:

```bash
# Generic format
ssh YOUR_USERID@eceterm1.uwaterloo.ca

# Example with oankit
ssh oankit@eceterm1.uwaterloo.ca
```

**Step 2:** From eceterm, connect to ecehadoop:

```bash
ssh ecehadoop.private.uwaterloo.ca
```

#### Method 3: Direct via ProxyJump (Off-Campus) ⭐ RECOMMENDED

Single command using SSH ProxyJump:

```bash
# Generic format
ssh -J YOUR_USERID@eceterm1.uwaterloo.ca YOUR_USERID@ecehadoop.private.uwaterloo.ca

# Example with oankit
ssh -J oankit@eceterm1.uwaterloo.ca oankit@ecehadoop.private.uwaterloo.ca
```

**Or add to your `~/.ssh/config`:**

```ssh-config
# ECE Terminal Server (Jump Host)
Host eceterm
    HostName eceterm1.uwaterloo.ca
    User oankit  # Replace with your user ID
    
# ECE Hadoop Cluster
Host ecehadoop
    HostName ecehadoop.private.uwaterloo.ca
    User oankit  # Replace with your user ID
    ProxyJump eceterm
```

Then simply:

```bash
ssh ecehadoop
```

### Important Connection Notes

1. **Master vs Worker Nodes:**
   - You connect to the **master node** (`ecehadoop.private.uwaterloo.ca`)
   - MPI distributes jobs to **worker nodes** (ecehadoop0-15)
   - Don't SSH directly to worker nodes for jobs

2. **ECE Terminal Servers:**
   - Available: `eceTerm1.uwaterloo.ca`, `eceTerm2.uwaterloo.ca`, `eceTerm3.uwaterloo.ca`
   - Used as jump hosts or for light terminal work
   - **NOT for heavy computation** - use them only to SSH to other servers

3. **X11 Forwarding:**
   - Not needed for MPI jobs
   - If you see: `Authorization required, but no authorization protocol specified`
   - Solution: Add `unset DISPLAY` to your `.cshrc` or run `setenv DISPLAY ''`

---

## Cluster Architecture

### Node Layout

```
┌─────────────────────────────────────────────────────────┐
│                    Master Node                          │
│          ecehadoop.private.uwaterloo.ca                 │
│  - Job submission                                       │
│  - File storage (home directory)                        │
│  - MPI coordination                                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Private Network
                   │
    ┌──────────────┼──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ ecehad  │  │ ecehad  │  │ ecehad  │  │ ecehad  │
│ oop0    │  │ oop1    │  │ oop2    │  │ oop3    │
│ 4 cores │  │ 4 cores │  │ 4 cores │  │ 4 cores │
└─────────┘  └─────────┘  └─────────┘  └─────────┘

     ... (12 more nodes) ...

┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ ecehad  │  │ ecehad  │  │ ecehad  │  │ ecehad  │
│ oop12   │  │ oop13   │  │ oop14   │  │ oop15   │
│ 4 cores │  │ 4 cores │  │ 4 cores │  │ 4 cores │
└─────────┘  └─────────┘  └─────────┘  └─────────┘

Total: 16 nodes × 4 cores = 64 cores
```

### Network Topology

```
Internet
    │
    ├─ UW Campus Network
    │       │
    │       ├─ eceTerm1.uwaterloo.ca (Jump Host)
    │       │
    │       └─ Private Network (.private.uwaterloo.ca)
    │               │
    │               ├─ ecehadoop (Master)
    │               │
    │               └─ ecehadoop0-15 (Workers)
    │
    └─ Off-Campus (VPN Required or Use Jump Host)
```

### MPI Slot Configuration

Each node has **4 slots** configured:

```
ecehadoop0.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop1.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop2.private.uwaterloo.ca slots=4 max-slots=4
...
ecehadoop15.private.uwaterloo.ca slots=4 max-slots=4
```

**What this means:**
- **slots=4**: Default number of processes per node
- **max-slots=4**: Maximum processes per node (hard limit)
- **Total capacity**: 16 nodes × 4 slots = 64 MPI ranks maximum

---

## Setting Up Your Environment

### 1. Initial SSH Access

Connect to the cluster:

```bash
# Generic format
ssh YOUR_USERID@ecehadoop.private.uwaterloo.ca

# Example with oankit (replace with your user ID)
ssh oankit@ecehadoop.private.uwaterloo.ca

# Or using jump host (off-campus)
ssh -J oankit@eceterm1.uwaterloo.ca oankit@ecehadoop.private.uwaterloo.ca
```

### 2. Shell Environment

**Default Shell:** `csh` or `tcsh` (not bash!)

Check your shell:

```bash
echo $SHELL
```

**Important Syntax Differences:**

| Operation | bash | csh/tcsh |
|-----------|------|----------|
| Set variable | `export VAR=value` | `setenv VAR value` |
| For loop | `for i in 1 2 3; do ... done` | `foreach i (1 2 3)` ... `end` |
| If statement | `if [ ... ]; then ... fi` | `if (...) then ... endif` |
| Run bash script | `bash script.sh` | `bash script.sh` (explicitly) |

**Tip:** For bash scripts, always run with `bash script.sh` or add shebang:

```bash
#!/bin/bash
```

### 3. Check Available Cores

Verify the number of cores on the master node:

```bash
nproc
```

Expected output: `4`

### 4. Python Setup

Check Python version:

```bash
python3 --version
```

Install required packages:

```bash
pip3 install --user mpi4py numpy numba opencv-python-headless scipy matplotlib pandas
```

**Note:** Use `--user` flag to install in your home directory.

### 5. Multi-Node SSH Setup (CRITICAL for MPI!)

For MPI to work across multiple nodes, passwordless SSH must be configured.

Your professor provides a script for this:

#### a. Get the Setup Script

From the course materials: `copy_ssh_public_key.sh`

Or create it manually:

```bash
#!/bin/bash
# Generate SSH key if not exists
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
fi

# Scan and add all worker nodes to known_hosts
for i in {0..15}; do
    ssh-keyscan -H ecehadoop$i.private.uwaterloo.ca >> ~/.ssh/known_hosts 2>/dev/null
done

# Copy public key to all worker nodes
for i in {0..15}; do
    echo "Setting up ecehadoop$i..."
    ssh-copy-id -i ~/.ssh/id_rsa.pub ecehadoop$i.private.uwaterloo.ca
done

echo "SSH setup complete!"
```

#### b. Run the Script

```bash
chmod +x copy_ssh_public_key.sh
bash copy_ssh_public_key.sh
```

#### c. Verify SSH Access

Test passwordless SSH to a worker node:

```bash
ssh ecehadoop0.private.uwaterloo.ca hostname
```

Should return: `ecehadoop0.private.uwaterloo.ca` without asking for password.

### 6. Create Hostfile

Create `mpi_ecehadoop_hosts` file:

```bash
cat > mpi_ecehadoop_hosts << 'EOF'
ecehadoop0.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop1.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop2.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop3.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop4.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop5.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop6.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop7.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop8.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop9.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop10.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop11.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop12.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop13.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop14.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop15.private.uwaterloo.ca slots=4 max-slots=4
EOF
```

---

## Running MPI Jobs

### MPI Command Structure

**Professor's Recommended Format:**

```bash
mpirun -np <NUM_RANKS> \
       --hostfile mpi_ecehadoop_hosts \
       --map-by slot \
       -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
       -x NUMBA_NUM_THREADS=1 \
       python3 your_script.py [arguments]
```

**Parameter Explanation:**

| Parameter | Meaning |
|-----------|---------|
| `-np <N>` | Number of MPI processes (ranks) |
| `--hostfile FILE` | File listing available nodes |
| `--map-by slot` | Distribute ranks by slots (fill nodes sequentially) |
| `-mca plm_rsh_args "..."` | SSH options for process launching |
| `-x VAR=value` | Export environment variable to all ranks |
| `NUMBA_NUM_THREADS=1` | CRITICAL: Prevent Numba oversubscription |

### Mapping Strategies

#### `--map-by slot` (Recommended)

Fills nodes sequentially:

```
4 ranks:  All on ecehadoop0
8 ranks:  4 on ecehadoop0, 4 on ecehadoop1
16 ranks: 4 per node on ecehadoop0-3
64 ranks: 4 per node on ecehadoop0-15
```

#### `--map-by node`

Round-robin across nodes:

```
4 ranks:  1 on ecehadoop0, 1 on ecehadoop1, 1 on ecehadoop2, 1 on ecehadoop3
8 ranks:  2 per node on ecehadoop0-3
```

**For MPI convolution, `--map-by slot` is usually better** (reduces inter-node communication for small jobs).

### Example: Running Your MPI Convolution

#### Single Test (4 ranks, 1 node)

```bash
mpirun -np 4 \
       --hostfile mpi_ecehadoop_hosts \
       --map-by slot \
       -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
       -x NUMBA_NUM_THREADS=1 \
       python3 mpi_conv.py --synthetic 2048 2048 --kernel-size 7 --padding same --stride 1 --csv results.csv
```

#### Scaling Experiment (4 to 64 ranks)

```bash
#!/bin/bash
unset DISPLAY

for n in 4 8 16 32 64; do
    echo "Running with $n ranks..."
    mpirun -np $n \
           --hostfile mpi_ecehadoop_hosts \
           --map-by slot \
           -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
           -x NUMBA_NUM_THREADS=1 \
           python3 mpi_conv.py --synthetic 4096 4096 --kernel-size 7 --padding same --stride 1 --csv scaling_results.csv
done

echo "Complete! Results:"
cat scaling_results.csv
```

### Checking Job Status

While job is running, you can check in another terminal:

```bash
# See processes on master node
top

# See which nodes are being used
mpirun -np 16 --hostfile mpi_ecehadoop_hosts hostname
```

### Canceling Jobs

```bash
# Find your processes
ps aux | grep your_userid | grep python3

# Kill all your Python processes
pkill -u your_userid python3

# Or kill specific process
kill <PID>
```

---

## Transferring Files

### From Local Machine to Cluster

#### Using `scp` (Windows/Linux/Mac)

**Single file:**

```bash
# Generic format
scp -J YOUR_USERID@eceterm1.uwaterloo.ca \
    local_file.py \
    YOUR_USERID@ecehadoop.private.uwaterloo.ca:~/

# Example with oankit
scp -J oankit@eceterm1.uwaterloo.ca \
    mpi_conv.py \
    oankit@ecehadoop.private.uwaterloo.ca:~/
```

**Entire directory:**

```bash
# Generic format
scp -r -J YOUR_USERID@eceterm1.uwaterloo.ca \
    ./your_project/ \
    YOUR_USERID@ecehadoop.private.uwaterloo.ca:~/

# Example with oankit - transferring entire assignment folder
scp -r -J oankit@eceterm1.uwaterloo.ca \
    "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment" \
    oankit@ecehadoop.private.uwaterloo.ca:~/mpi_project/
```

#### Using WinSCP (Windows GUI)

1. Install WinSCP from https://winscp.net/
2. Configure connection:
   - **Protocol**: SFTP
   - **Host name**: `ecehadoop.private.uwaterloo.ca`
   - **User name**: `oankit` (replace with your user ID)
   - Click **Advanced** → **Connection** → **Tunnel**
   - ✅ Check "Connect through SSH tunnel"
   - **Tunnel host name**: `eceterm1.uwaterloo.ca`
   - **Tunnel user name**: `oankit` (replace with your user ID)
   - Click **OK**, then **Save**, then **Login**

#### From Cluster to Local Machine

```bash
# Generic format
scp -J YOUR_USERID@eceterm1.uwaterloo.ca \
    YOUR_USERID@ecehadoop.private.uwaterloo.ca:~/results.csv \
    ./local_directory/

# Example with oankit - downloading results
scp -J oankit@eceterm1.uwaterloo.ca \
    oankit@ecehadoop.private.uwaterloo.ca:~/cluster_results.csv \
    "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment\"
```

### Within Cluster

Once connected to ecehadoop:

```bash
# Copy to your home directory
cp /path/to/file ~/

# Copy between directories
cp -r source_dir/ dest_dir/
```

---

## Best Practices

### 1. Resource Usage

**Be a Good Cluster Citizen:**

- **Test small first:** Run with 4 ranks before scaling to 64
- **Use appropriate resources:** Don't request 64 ranks for a 10-second job
- **Check cluster load:** Use `top` or `uptime` before large jobs
- **Clean up:** Remove large temporary files after experiments

**Avoid Oversubscription:**

```bash
# BAD: 64 ranks with Numba threading (256 threads competing for 64 cores!)
mpirun -np 64 python3 script.py

# GOOD: 64 ranks with NUMBA_NUM_THREADS=1 (64 threads for 64 cores)
mpirun -np 64 -x NUMBA_NUM_THREADS=1 python3 script.py
```

### 2. File Management

**Home Directory:**
- Your home directory is shared across all nodes
- Use it for code, scripts, and results
- Don't store huge temporary files

**Temporary Files:**
- Use `/tmp` on each node for temporary per-node storage
- Clean up after jobs complete

**Result Collection:**
- Use CSV files for results (easy to transfer and analyze)
- Run analysis locally on your machine

### 3. Debugging Workflow

```
1. Develop/test locally (single process)
   └─> python3 mpi_conv.py --synthetic 512 512 ...

2. Test MPI locally (4 processes on your laptop)
   └─> mpirun -np 4 python3 mpi_conv.py --synthetic 1024 1024 ...

3. Test on cluster (4 processes, 1 node)
   └─> mpirun -np 4 --hostfile mpi_ecehadoop_hosts ...

4. Scale on cluster (8, 16, 32, 64 processes)
   └─> for n in 8 16 32 64; do mpirun -np $n ...
```

### 4. Monitoring Performance

**Add timing to your code:**

```python
import time

start = time.time()
# ... your code ...
end = time.time()

if rank == 0:
    print(f"Total time: {end - start:.2f}s")
```

**Log to CSV for analysis:**

```python
import csv

results = {
    'ranks': size,
    'image_size': H,
    't_total': total_time,
    't_comp': comp_time,
    't_comm': comm_time,
}

if rank == 0:
    with open('results.csv', 'a') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:  # Empty file
            writer.writeheader()
        writer.writerow(results)
```

### 5. Shell Scripting Tips

**Always specify bash for bash scripts:**

```bash
#!/bin/bash
# Your script here
```

**Run bash scripts explicitly:**

```bash
bash my_script.sh
```

**For csh/tcsh scripts:**

```csh
#!/bin/csh
# Use csh syntax
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "ssh: Could not resolve hostname"

**Error:**
```
ssh: Could not resolve hostname ecehadoop.private.uwaterloo.ca
```

**Solutions:**
- **Off-campus:** Use VPN or jump host (`-J eceterm1.uwaterloo.ca`)
- **On-campus:** Check network connection
- **Typo:** Verify hostname spelling

#### 2. "There are not enough slots available"

**Error:**
```
There are not enough slots available in the system to satisfy the 8 slots
```

**Solutions:**
- Check hostfile exists and is readable: `cat mpi_ecehadoop_hosts`
- Verify slot count: 16 nodes × 4 slots = 64 max
- Reduce number of ranks: Use `-np` ≤ 64

#### 3. "ORTE does not know how to route a message"

**Error:**
```
ORTE does not know how to route a message to the specified daemon
```

**Causes:**
- Passwordless SSH not configured
- Worker nodes not reachable

**Solutions:**
- Run `bash copy_ssh_public_key.sh`
- Test SSH: `ssh ecehadoop0.private.uwaterloo.ca hostname`
- Check hostfile has correct format

#### 4. "Permission denied (publickey)"

**Error:**
```
Permission denied (publickey,gssapi-keyex,gssapi-with-mic)
```

**Solutions:**
- Generate SSH key: `ssh-keygen -t rsa`
- Copy to nodes: `ssh-copy-id ecehadoop0.private.uwaterloo.ca`
- Or run professor's setup script

#### 5. "Authorization required, but no authorization protocol specified"

**Error:**
```
Authorization required, but no authorization protocol specified
Error: Can't open display
```

**Solution:**
```bash
# In csh/tcsh:
setenv DISPLAY ''

# Or in bash:
export DISPLAY=

# Or in your code/script:
unset DISPLAY
```

**Why:** X11 display warning (can be ignored for command-line jobs)

#### 6. Slow Performance / Oversubscription

**Symptoms:**
- Job slower with more ranks
- CPU shows >100% usage per core
- High context switching

**Solution:**
```bash
# Always set NUMBA_NUM_THREADS=1 when using MPI + Numba
mpirun -np 16 -x NUMBA_NUM_THREADS=1 python3 script.py
```

**Why:** Numba default is to use all cores per process, causing 64 ranks to spawn 256+ threads.

#### 7. "Command not found" in Shell

**Issue:** You wrote a bash script but cluster uses csh/tcsh

**Solution:**
```bash
# Run bash scripts explicitly
bash my_script.sh

# Or check shell and switch
echo $SHELL
bash  # Start bash shell
```

#### 8. "No such file or directory" for Python Packages

**Error:**
```
ModuleNotFoundError: No module named 'mpi4py'
```

**Solution:**
```bash
# Install with --user flag
pip3 install --user mpi4py numpy numba

# Check installation
python3 -c "import mpi4py; print(mpi4py.__version__)"
```

#### 9. Job Hangs / Doesn't Complete

**Possible causes:**
- Waiting for input (code expects user input)
- Deadlock in MPI communication
- Infinite loop
- Network issue

**Solutions:**
- Check code for `input()` calls
- Review MPI Send/Recv matching
- Add timeout or Ctrl+C and check logs
- Add print statements to track progress

#### 10. "Badly placed ()'s" Error

**Error:**
```
Badly placed ()'s
```

**Cause:** Trying to run bash syntax in csh/tcsh shell

**Solution:**
```bash
# Run bash scripts with bash
bash script.sh

# Or switch to bash
bash
source ~/.bashrc  # If you have one
```

---

## Quick Reference Card

**Note:** Replace `oankit` with your UW user ID in all commands below.

### Connection

```bash
# Off-campus (via jump host) - RECOMMENDED
ssh -J oankit@eceterm1.uwaterloo.ca oankit@ecehadoop.private.uwaterloo.ca

# On-campus (direct)
ssh oankit@ecehadoop.private.uwaterloo.ca
```

### File Transfer

```bash
# Upload single file
scp -J oankit@eceterm1.uwaterloo.ca \
    local_file \
    oankit@ecehadoop.private.uwaterloo.ca:~/

# Upload directory
scp -r -J oankit@eceterm1.uwaterloo.ca \
    ./Assignment \
    oankit@ecehadoop.private.uwaterloo.ca:~/

# Download file
scp -J oankit@eceterm1.uwaterloo.ca \
    oankit@ecehadoop.private.uwaterloo.ca:~/results.csv \
    ./
```

### MPI Job

```bash
mpirun -np 16 \
       --hostfile mpi_ecehadoop_hosts \
       --map-by slot \
       -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
       -x NUMBA_NUM_THREADS=1 \
       python3 mpi_conv.py --synthetic 4096 4096 --kernel-size 7 --padding same --stride 1
```

### Useful Commands

```bash
# Check cores on current node
nproc

# Check running processes
top

# Check MPI version
mpirun --version

# Test hostfile (should show 16 different hostnames)
mpirun -np 16 --hostfile mpi_ecehadoop_hosts hostname

# Kill your Python jobs (replace oankit with your user ID)
pkill -u oankit python3

# Check if your jobs are running
ps aux | grep oankit | grep python3
```

---

## Additional Resources

### Course Materials

Reference link from professor:
- https://learn.uwaterloo.ca/d2l/le/content/1171488/viewContent/6124691/View

### ECE Linux Servers Status

Live status page:
- https://ece.uwaterloo.ca/~epraetze/servers/servers-all.html

### SSH Authman Guide

Official documentation:
- https://uwaterloo.atlassian.net/wiki/spaces/ISTKB/pages/1551728653/How+to+use+SSH+Authman+-+for+users

### OpenMPI Documentation

For advanced MPI options:
- https://www.open-mpi.org/doc/

---

## Summary: Getting Started Checklist

**Replace `oankit` with your UW user ID in all commands.**

- [ ] Connect to cluster: `ssh -J oankit@eceterm1.uwaterloo.ca oankit@ecehadoop.private.uwaterloo.ca`
- [ ] Check Python: `python3 --version`
- [ ] Install packages: `pip3 install --user mpi4py numpy numba opencv-python-headless scipy matplotlib pandas`
- [ ] Run SSH setup script: `bash copy_ssh_public_key.sh`
- [ ] Create hostfile: `mpi_ecehadoop_hosts` (16 nodes, 4 slots each)
- [ ] Test SSH to worker: `ssh ecehadoop0.private.uwaterloo.ca hostname`
- [ ] Transfer code: `scp -r -J oankit@eceterm1.uwaterloo.ca ./Assignment oankit@ecehadoop.private.uwaterloo.ca:~/`
- [ ] Test MPI (4 ranks): `mpirun -np 4 --hostfile mpi_ecehadoop_hosts -x NUMBA_NUM_THREADS=1 python3 mpi_conv.py --synthetic 2048 2048 --kernel-size 7 --padding same --stride 1`
- [ ] Run scaling experiments: 8, 16, 32, 64 ranks with 4096×4096 images
- [ ] Download results: `scp -J oankit@eceterm1.uwaterloo.ca oankit@ecehadoop.private.uwaterloo.ca:~/cluster_results.csv ./`

---

## Real-World Example: Complete Workflow

Here's the actual workflow that was successfully used to run experiments on the cluster:

### Step 1: Connect to Cluster (Off-Campus)

```bash
# From Windows local machine
ssh -J oankit@eceterm1.uwaterloo.ca oankit@ecehadoop.private.uwaterloo.ca
```

### Step 2: Setup Environment

```bash
# Check system
nproc  # Output: 4 (master node has 4 cores)
python3 --version

# Install packages
pip3 install --user mpi4py numpy numba opencv-python-headless scipy matplotlib pandas

# Setup passwordless SSH
bash copy_ssh_public_key.sh  # From professor's materials

# Verify SSH access to workers
ssh ecehadoop0.private.uwaterloo.ca hostname
# Output: ecehadoop0.private.uwaterloo.ca
```

### Step 3: Transfer Code from Windows

```bash
# From Windows PowerShell/Command Prompt (NOT on cluster)
scp -r -J oankit@eceterm1.uwaterloo.ca ^
    "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment" ^
    oankit@ecehadoop.private.uwaterloo.ca:~/

# On Linux/Mac, use backslash for line continuation instead of caret (^)
```

### Step 4: Create Hostfile

```bash
# Back on cluster terminal
cd ~/Assignment

cat > mpi_ecehadoop_hosts << 'EOF'
ecehadoop0.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop1.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop2.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop3.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop4.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop5.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop6.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop7.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop8.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop9.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop10.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop11.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop12.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop13.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop14.private.uwaterloo.ca slots=4 max-slots=4
ecehadoop15.private.uwaterloo.ca slots=4 max-slots=4
EOF
```

**Note:** Type `EOF` exactly on a new line and press Enter to complete the heredoc.

### Step 5: Test MPI (Single Node)

```bash
# Disable X11 (avoids display warning)
unset DISPLAY

# Test with 4 ranks on 1 node
mpirun -np 4 \
       --hostfile mpi_ecehadoop_hosts \
       --map-by slot \
       -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
       -x NUMBA_NUM_THREADS=1 \
       python3 mpi_conv.py --synthetic 2048 2048 --kernel-size 7 --padding same --stride 1 --csv test_results.csv

# Check results
cat test_results.csv
```

### Step 6: Run Scaling Experiments

```bash
# Create experiment script
cat > run_scaling_experiments.sh << 'EOF'
#!/bin/bash
unset DISPLAY

echo "Starting MPI Convolution Scaling Experiments"
echo "============================================="

for n in 4 8 16 32 64; do
    echo ""
    echo "Running with $n ranks on $(($n / 4)) nodes..."
    mpirun -np $n \
           --hostfile mpi_ecehadoop_hosts \
           --map-by slot \
           -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
           -x NUMBA_NUM_THREADS=1 \
           python3 mpi_conv.py --synthetic 4096 4096 --kernel-size 7 --padding same --stride 1 --csv cluster_results.csv
    
    echo "Completed $n ranks"
done

echo ""
echo "All experiments complete!"
echo "Results summary:"
cat cluster_results.csv
EOF

# Make executable and run
chmod +x run_scaling_experiments.sh
bash run_scaling_experiments.sh
```

### Step 7: Download Results

```bash
# From Windows local machine (NOT on cluster)
scp -J oankit@eceterm1.uwaterloo.ca ^
    oankit@ecehadoop.private.uwaterloo.ca:~/Assignment/cluster_results.csv ^
    "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment\"

# Verify file downloaded
dir "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment\cluster_results.csv"
```

### Step 8: Analyze Locally

```bash
# Back on local Windows machine
cd "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment"
python create_plots.py
```

### Results Summary

**Successful Experiments:**
- ✅ 4 ranks (1 node): ~1.38s total
- ✅ 8 ranks (2 nodes): Working but slower than expected
- ✅ 16 ranks (4 nodes): ~2.01s total
- ✅ 32 ranks (8 nodes): Communication overhead visible
- ✅ 64 ranks (16 nodes): All nodes utilized

**Key Findings:**
- Problem size (4096×4096) too small for good scaling
- Communication overhead dominates at higher rank counts
- Need to set `NUMBA_NUM_THREADS=1` to avoid oversubscription
- Works best with 4-16 ranks for this problem size

---

## Questions or Issues?

If you encounter problems:

1. Check this guide's Troubleshooting section
2. Review professor's course materials
3. Ask on course forum or contact TA
4. Check ECE server status page

**Good luck with your distributed computing experiments!** 🚀

