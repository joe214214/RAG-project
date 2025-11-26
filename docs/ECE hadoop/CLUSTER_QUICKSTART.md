# ECE Hadoop Cluster - Quick Start Guide

## Before You Start

### 1. Verify Cluster Access

**Test connection:**
```bash
# From your Windows machine (PowerShell or Git Bash)
ssh <your_uwaterloo_id>@ecehadoop.uwaterloo.ca
```

**If connection fails:**
- ❌ "Permission denied" → You don't have access yet. Email Dr. Reza.
- ❌ "Connection refused/timeout" → Try VPN or use eceterm1 as jump host:
  ```bash
  ssh <your_id>@eceterm1.uwaterloo.ca
  # Then from eceterm1:
  ssh ecehadoop
  ```

**If connection succeeds:**
- ✅ You're in! Note what you see (hostname, welcome message)

---

## Initial Setup (First Time Only)

### Step 1: Check Environment

Once logged into the cluster:

```bash
# Check hostname
hostname
# Should see: ecehadoop or similar

# Check Python
python3 --version
# Need: 3.8 or higher

# Check MPI
which mpirun
mpirun --version
# Should show MPICH or OpenMPI

# Check if there's a job scheduler
which sbatch    # Slurm
which qsub      # PBS/Torque
# If nothing found, you'll use mpirun directly
```

### Step 2: Create Working Directory

```bash
# Create your workspace
mkdir -p ~/ece750_assignment
cd ~/ece750_assignment
```

### Step 3: Transfer Your Code

**Option A: Using SCP from Windows**
```powershell
# From Windows PowerShell (on your laptop)
cd "C:\Users\omara\Documents\UWaterloo\ECE 750\Assignment"

# Copy all Python files
scp *.py <your_id>@ecehadoop.uwaterloo.ca:~/ece750_assignment/

# Copy setup script
scp cluster_setup.sh <your_id>@ecehadoop.uwaterloo.ca:~/ece750_assignment/
```

**Option B: Using Git (if you have a repo)**
```bash
# On the cluster
cd ~/ece750_assignment
git clone <your_repo_url> .
```

**Option C: Copy-paste small files**
```bash
# On the cluster, create files manually
nano mpi_conv.py
# Paste content, save with Ctrl+X, Y, Enter
```

### Step 4: Run Setup Script

```bash
cd ~/ece750_assignment
chmod +x cluster_setup.sh
bash cluster_setup.sh
```

This will:
- Create a Python virtual environment
- Install dependencies (mpi4py, numpy, numba, etc.)
- Test MPI setup
- Create output directories

### Step 5: Quick Sanity Test

```bash
# Activate environment
source ~/ece750_assignment/venv/bin/activate

# Test with 4 processes on single node
mpirun -n 4 python mpi_conv.py \
  --synthetic 512 512 \
  --kernel-size 5 \
  --padding same \
  --stride 1 \
  --px 2 --py 2 \
  --check

# Should see: validation results and timing
```

**If this works, you're ready to run experiments!**

---

## Understanding the Cluster

### Cluster Architecture

- **Master node**: ecehadoop (where you log in)
- **Worker nodes**: ecehadoop0-ecehadoop15 (16 nodes)
- **Cores per node**: 4 cores
- **Total cores**: 16 nodes × 4 = 64 cores

### Running MPI Jobs

**Single node (up to 4 cores):**
```bash
mpirun -n 4 python mpi_conv.py --synthetic 1024 1024 --kernel-size 7 --px 2 --py 2
```

**Multiple nodes (e.g., 16 cores across 4 nodes):**
```bash
# Create hostfile
cat > hostfile << EOF
ecehadoop0.private.uwaterloo.ca slots=4
ecehadoop1.private.uwaterloo.ca slots=4
ecehadoop2.private.uwaterloo.ca slots=4
ecehadoop3.private.uwaterloo.ca slots=4
EOF

# Run with hostfile
mpirun -n 16 --hostfile hostfile python mpi_conv.py \
  --synthetic 2048 2048 --kernel-size 7 --px 4 --py 4
```

**Note:** You may need to add SSH options:
```bash
mpirun -n 16 --hostfile hostfile \
  -mca plm_rsh_args "-o StrictHostKeyChecking=no" \
  python mpi_conv.py ...
```

---

## Troubleshooting

### "No module named 'mpi4py'"
```bash
# Make sure virtual environment is activated
source ~/ece750_assignment/venv/bin/activate

# Reinstall if needed
pip install mpi4py
```

### "Permission denied" when accessing worker nodes
```bash
# Make sure SSH keys are set up
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Test SSH to worker node
ssh -o StrictHostKeyChecking=no ecehadoop0.private.uwaterloo.ca echo "OK"
```

### "Cannot allocate memory" errors
- Reduce batch size: `--batch 1`
- Reduce image size
- Use fewer processes per node

### Jobs taking too long / cluster busy
```bash
# Check cluster load
ssh ecehadoop0.private.uwaterloo.ca "uptime"

# See who else is using the cluster
w
```

---

## What to Do After Setup

Once setup is complete:

1. ✅ Run small test (4 cores, 512×512 image)
2. ✅ Run medium test (16 cores, 2048×2048 image)
3. ✅ Run full experiments (see `run_experiments.sh`)
4. ✅ Download results: `scp <id>@ecehadoop:~/ece750_assignment/results/*.csv .`
5. ✅ Analyze and plot results

---

## Quick Reference

**Connect to cluster:**
```bash
ssh <your_id>@ecehadoop.uwaterloo.ca
```

**Activate environment:**
```bash
cd ~/ece750_assignment
source venv/bin/activate
```

**Run experiment:**
```bash
bash run_experiments.sh
```

**Download results:**
```bash
# From your Windows machine
scp <your_id>@ecehadoop.uwaterloo.ca:~/ece750_assignment/results/*.csv ./results/
```

**Monitor job:**
```bash
# Watch output in real-time
tail -f logs/experiment_YYYYMMDD_HHMMSS.log
```

---

## Need Help?

1. Check cluster status: https://ece.uwaterloo.ca/Nexus/arbeau/clients/
2. Email instructor: tahsin.reza@uwaterloo.ca
3. Ask your teammates
4. Check assignment PDF page 22-23 for cluster details

