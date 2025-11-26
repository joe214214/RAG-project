# ECE Hadoop Cluster - Quick Setup Guide

## Your Access Info
- **UserID**: oankit
- **Hostname**: ecehadoop.private.uwaterloo.ca
- **You have access!**

## Step 1: Connect to Cluster

```bash
ssh oankit@ecehadoop.private.uwaterloo.ca
```

If you're off-campus, you may need:
1. UWaterloo VPN, OR
2. Jump through eceterm1 first:
```bash
ssh oankit@eceterm1.uwaterloo.ca
# Then from eceterm1:
ssh ecehadoop
```

## Step 2: Transfer Your Code

From your Windows machine (PowerShell):

```powershell
# Navigate to your assignment directory
cd "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment"

# Copy all Python files to cluster
scp *.py oankit@ecehadoop.private.uwaterloo.ca:~/ece750_assignment/
```

Or transfer one at a time:
```powershell
scp mpi_conv.py oankit@ecehadoop.private.uwaterloo.ca:~/
scp domain.py oankit@ecehadoop.private.uwaterloo.ca:~/
scp comm.py oankit@ecehadoop.private.uwaterloo.ca:~/
# ... etc
```

## Step 3: Setup Environment on Cluster

Once logged into ecehadoop:

```bash
# Create working directory
mkdir -p ~/ece750_assignment
cd ~/ece750_assignment

# Check Python version
python3 --version

# Install dependencies (user-level)
pip3 install --user mpi4py numpy numba scipy opencv-python

# Check MPI
which mpirun
mpirun --version
```

## Step 4: Test MPI

Quick test with 4 cores:

```bash
mpirun -n 4 python3 mpi_conv.py \
  --synthetic 1024 1024 \
  --kernel-size 7 \
  --padding same \
  --stride 1 \
  --px 2 --py 2 \
  --check
```

## Step 5: Run Scaling Experiments

### Strong Scaling (fixed problem, varying cores)

```bash
#!/bin/bash
# strong_scaling.sh

for n in 1 2 4 8 16 32 64; do
    echo "Running with $n ranks..."
    mpirun -n $n python3 mpi_conv.py \
      --synthetic 4096 4096 \
      --kernel-size 7 \
      --padding same \
      --stride 1 \
      --csv strong_scaling.csv \
      --baseline 10.0
done
```

Save as `strong_scaling.sh`, then:
```bash
chmod +x strong_scaling.sh
./strong_scaling.sh
```

### Weak Scaling (problem grows with cores)

```bash
#!/bin/bash
# weak_scaling.sh

# 1 rank: 1024x1024
mpirun -n 1 python3 mpi_conv.py --synthetic 1024 1024 --kernel-size 7 --csv weak_scaling.csv

# 4 ranks: 2048x2048
mpirun -n 4 python3 mpi_conv.py --synthetic 2048 2048 --kernel-size 7 --csv weak_scaling.csv

# 16 ranks: 4096x4096
mpirun -n 16 python3 mpi_conv.py --synthetic 4096 4096 --kernel-size 7 --csv weak_scaling.csv

# 64 ranks: 8192x8192
mpirun -n 64 python3 mpi_conv.py --synthetic 8192 8192 --kernel-size 7 --csv weak_scaling.csv
```

## Step 6: Download Results

From Windows:

```powershell
# Download CSV results
scp oankit@ecehadoop.private.uwaterloo.ca:~/ece750_assignment/*.csv ./

# Or specific file
scp oankit@ecehadoop.private.uwaterloo.ca:~/ece750_assignment/strong_scaling.csv ./
```

## Useful Commands

```bash
# Check if your job is running
top

# Monitor resource usage
htop

# See cluster load
uptime

# Kill a job if needed
pkill -u oankit python3
```

## MPI Run Scripts

Dr. Reza mentioned: https://learn.uwaterloo.ca/d2l/le/content/1171488/viewContent/6124691/View

Check that link for official cluster submission scripts.

## Quick Reference

**Connect:**
```bash
ssh oankit@ecehadoop.private.uwaterloo.ca
```

**Transfer code:**
```powershell
scp *.py oankit@ecehadoop.private.uwaterloo.ca:~/ece750_assignment/
```

**Run experiment:**
```bash
cd ~/ece750_assignment
mpirun -n 8 python3 mpi_conv.py --synthetic 2048 2048 --kernel-size 7 --check
```

**Download results:**
```powershell
scp oankit@ecehadoop.private.uwaterloo.ca:~/ece750_assignment/*.csv ./
```

---

You're all set! Start by connecting and transferring your code.

