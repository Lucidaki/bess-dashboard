# BESS Dashboard Deployment Guide

## Overview

This guide covers three hosting options for deploying the BESS Dashboard:

1. **Streamlit Cloud** (Recommended for quick private deployment)
2. **Hugging Face Spaces** (Best for public demos/portfolio)
3. **Oracle Cloud Free Tier** (Best for production deployment)

---

## Option 1: Streamlit Cloud (RECOMMENDED)

**Best for:** Private internal deployment with minimal setup

### Prerequisites
- GitHub account
- Streamlit Cloud account (free - sign up at [share.streamlit.io](https://share.streamlit.io))
- Code pushed to GitHub repository

### Step 1: Prepare Repository

Your repository must be public or you need Streamlit Cloud's GitHub app installed for private repos.

**Check your current Git status:**
```bash
cd c:/repos/bess-dashboard/bess-dashboard-1
git status
```

**Commit and push latest changes:**
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Configure deployment:**
   - **Repository:** Select your repo (e.g., `yourusername/bess-dashboard-1`)
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom URL (e.g., `bess-dashboard-yourname`)

5. **Advanced settings (click dropdown):**
   - **Python version:** 3.10
   - **Make app private:** ✅ ENABLE THIS (requires email authentication)

6. **Click "Deploy!"**

Deployment takes 3-5 minutes. You'll see build logs in real-time.

### Step 3: Configure Access Control (Private Apps)

Once deployed:

1. **Click "Settings" (⚙️) → "Sharing"**
2. **Add authorized emails:**
   - Add emails of Finance team members
   - Add emails of O&M team members
   - Only these users can access the dashboard

3. **Access URL:** `https://bess-dashboard-yourname.streamlit.app`

### Step 4: Test the Deployment

1. Open the app URL in incognito mode
2. You should be prompted to enter email
3. Enter an authorized email → receive magic link → access dashboard
4. Upload test CSV files to verify end-to-end workflow

### Maintenance

**Update the app:**
```bash
# Make code changes locally
git add .
git commit -m "Update feature X"
git push origin main
# App automatically redeploys within 2 minutes
```

**View logs:**
- Go to [share.streamlit.io](https://share.streamlit.io) → Manage app → Logs

**Resource limits on Free tier:**
- 1 GB RAM
- 1 CPU core
- Apps sleep after 7 days of inactivity (30-second wake time)
- 3 private apps maximum

---

## Option 2: Hugging Face Spaces

**Best for:** Public portfolio projects or demos

### Prerequisites
- Hugging Face account (free - sign up at [huggingface.co](https://huggingface.co))
- Git installed locally

### Step 1: Create Hugging Face Space

1. **Go to [huggingface.co](https://huggingface.co)**
2. **Click your profile → "New Space"**
3. **Configure Space:**
   - **Space name:** `bess-dashboard`
   - **License:** Apache 2.0 (or your choice)
   - **Select SDK:** Streamlit
   - **Space hardware:** CPU basic (free)
   - **Visibility:** Public (or Private with $5/month)

4. **Click "Create Space"**

### Step 2: Prepare Deployment Files

**Create `README.md` for the Space:**
```bash
cd c:/repos/bess-dashboard/bess-dashboard-1
```

Create `README.md` with this content:
```markdown
---
title: BESS Arbitrage Optimization Dashboard
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
---

# BESS Dashboard

Battery Energy Storage System (BESS) arbitrage optimization and performance analytics platform.

## Features
- 📊 Data Quality Analysis
- ⚡ MILP Optimization
- 💰 Finance KPIs (Market Capture, Revenue Variance)
- 🔧 O&M KPIs (Availability, Cycle Utilization)
- 📈 Interactive Visualizations

## Usage
1. Navigate to "Data Upload" page
2. Upload SCADA CSV (timestamp, power_mw, soc_percent)
3. Upload Market Price CSV (timestamp, price_gbp_mwh, market_type)
4. Run optimization
5. View results across Finance, O&M, and Optimization pages
```

### Step 3: Push to Hugging Face

```bash
# Add Hugging Face remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/bess-dashboard

# Push to Hugging Face
git push hf main
```

**If prompted for credentials:**
- Username: Your Hugging Face username
- Password: Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 4: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/bess-dashboard`
2. Click "Logs" to see build progress
3. Deployment takes 5-10 minutes
4. App will be live at: `https://YOUR_USERNAME-bess-dashboard.hf.space`

### Maintenance

**Update the Space:**
```bash
git add .
git commit -m "Update feature"
git push hf main
```

**Resource limits on Free tier:**
- 16 GB RAM
- 8 CPU cores
- No sleep/cold starts
- ⚠️ Public by default (private requires $5/month)

---

## Option 3: Oracle Cloud Free Tier

**Best for:** Production deployment with always-on availability

### Prerequisites
- Oracle Cloud account (free - sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/))
- SSH key pair generated

### Step 1: Create Compute Instance

Refer to the detailed Oracle Cloud deployment guide you received earlier. Key steps:

1. **Navigate to Compute → Instances**
2. **Create instance:**
   - Name: `bess-dashboard`
   - Image: Ubuntu 22.04
   - Shape: VM.Standard.E2.1.Micro (Always Free eligible)
   - Network: Configure public IP
3. **Configure security rules:**
   - Allow TCP port 22 (SSH)
   - Allow TCP port 8501 (Streamlit)

### Step 2: Install Dependencies

SSH into your instance:
```bash
ssh -i ~/.ssh/oracle_cloud ubuntu@<YOUR_INSTANCE_IP>
```

Install system packages:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git build-essential
```

### Step 3: Deploy Application

Clone repository:
```bash
cd /opt
sudo mkdir bess-dashboard
sudo chown ubuntu:ubuntu bess-dashboard
cd bess-dashboard
git clone https://github.com/YOUR_USERNAME/bess-dashboard-1.git .
```

Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Create Systemd Service

Create service file:
```bash
sudo nano /etc/systemd/system/bess-dashboard.service
```

Add this content:
```ini
[Unit]
Description=BESS Dashboard Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/bess-dashboard
Environment="PATH=/opt/bess-dashboard/venv/bin"
ExecStart=/opt/bess-dashboard/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable bess-dashboard
sudo systemctl start bess-dashboard
sudo systemctl status bess-dashboard
```

### Step 5: Access Dashboard

Open browser: `http://<YOUR_INSTANCE_IP>:8501`

### Maintenance

**Update application:**
```bash
cd /opt/bess-dashboard
git pull origin main
sudo systemctl restart bess-dashboard
```

**View logs:**
```bash
sudo journalctl -u bess-dashboard -f
```

**Resource limits on Free tier:**
- 1 OCPU (equivalent to 2 vCPUs)
- 1 GB RAM
- 200 GB block storage
- 10 TB outbound data transfer/month
- Always-on (no cold starts)

---

## Comparison Matrix

| Feature | Streamlit Cloud | Hugging Face | Oracle Cloud |
|---------|----------------|--------------|--------------|
| **Setup Time** | 5 minutes | 10 minutes | 30 minutes |
| **Free Tier Resources** | 1 GB RAM, 1 CPU | 16 GB RAM, 8 CPU | 1 GB RAM, 1 OCPU |
| **Cold Starts** | Yes (after 7 days) | No | No |
| **Private by Default** | ✅ Yes (3 apps) | ❌ No ($5/month) | ✅ Yes |
| **Email Auth** | ✅ Built-in | ❌ No | Manual setup |
| **Auto-Deploy from Git** | ✅ Yes | ✅ Yes | ❌ Manual |
| **Custom Domain** | ❌ No (paid tier) | ❌ No | ✅ Yes |
| **SSH Access** | ❌ No | ❌ No | ✅ Yes |
| **Best For** | Private demos | Public portfolio | Production |

---

## Recommendation

### For Your Use Case (BESS Commercial Tool):

**Primary Deployment:** Streamlit Cloud (Private App)
- ✅ 5-minute deployment
- ✅ Email-based authentication
- ✅ Free private hosting
- ✅ Perfect for Finance/O&M stakeholder demos

**Secondary Deployment:** Oracle Cloud
- Use when stakeholders adopt the tool for daily operations
- Always-on availability
- No resource limits for CSV-based workflows

**Not Recommended:** Hugging Face Spaces
- Public by default (not suitable for commercial SCADA/price data)
- Private tier costs $5/month (Streamlit Cloud offers 3 private apps free)

---

## Troubleshooting

### Streamlit Cloud

**Issue:** "App is too large to deploy"
- **Solution:** Check if `data/` directory is pushed to Git. Add to `.gitignore`:
  ```
  data/raw/*
  data/canonical/*
  data/optimization_results/*
  ```

**Issue:** "Module not found"
- **Solution:** Verify `requirements.txt` includes all dependencies
- Check for version conflicts with `pip freeze > requirements_full.txt`

**Issue:** "Out of memory"
- **Solution:**
  - Process smaller datasets
  - Upgrade to Streamlit Cloud paid tier (more RAM)
  - Switch to Oracle Cloud

### Hugging Face Spaces

**Issue:** "Build failed"
- **Solution:** Check "Logs" tab for specific error
- Verify `README.md` has correct YAML front matter
- Ensure `sdk: streamlit` and `app_file: app.py` are set

**Issue:** "App shows 404"
- **Solution:** Verify `app.py` is in repository root, not subfolder

### Oracle Cloud

**Issue:** "Cannot connect via SSH"
- **Solution:**
  - Verify security list allows TCP port 22
  - Check VCN subnet security rules
  - Verify SSH key permissions: `chmod 600 ~/.ssh/oracle_cloud`

**Issue:** "Dashboard not accessible on port 8501"
- **Solution:**
  - Add ingress rule for TCP port 8501 in security list
  - Configure Ubuntu firewall:
    ```bash
    sudo ufw allow 8501
    sudo ufw enable
    ```

**Issue:** "Service fails to start"
- **Solution:** Check logs:
  ```bash
  sudo journalctl -u bess-dashboard -n 50
  ```
- Common causes:
  - Wrong Python path in service file
  - Missing dependencies (reinstall requirements.txt)
  - Port already in use (check with `sudo netstat -tlnp | grep 8501`)

---

## Security Best Practices

### All Platforms
- ✅ **Never commit** `.env` files or API keys to Git
- ✅ Add `.gitignore` for:
  ```
  .env
  data/raw/*
  data/canonical/*
  *.pyc
  __pycache__/
  ```
- ✅ Use Streamlit secrets management for sensitive data

### Streamlit Cloud
- ✅ Enable private app mode
- ✅ Use email allowlist (not public link sharing)
- ✅ Add secrets in Settings → Secrets (not in code)

### Oracle Cloud
- ✅ Use SSH keys (not passwords)
- ✅ Enable only required ports (22, 8501)
- ✅ Keep system updated: `sudo apt-get update && sudo apt-get upgrade`
- ✅ Consider setting up HTTPS with Let's Encrypt (nginx reverse proxy)

---

## Cost Breakdown

### Free Tier Limits

| Platform | Free Tier | Paid Tier (Starting) |
|----------|-----------|----------------------|
| **Streamlit Cloud** | 3 private apps, unlimited public | $20/month (5 private apps, more resources) |
| **Hugging Face** | Unlimited public spaces | $5/month (private space) |
| **Oracle Cloud** | 2 VMs Always Free | Pay-as-you-go (start at $0.01/hr) |

### Recommendation by Budget

- **$0/month:** Streamlit Cloud (3 private apps covers your MVP)
- **$5/month:** Hugging Face private space (if you need public demo + private version)
- **$0/month (production):** Oracle Cloud Always Free (no time limits)

---

## Next Steps

1. **Deploy to Streamlit Cloud** (takes 5 minutes) following Step 1
2. **Test with stakeholders** (Finance + O&M teams)
3. **Collect feedback** on performance and features
4. **Scale to Oracle Cloud** when moving to production (Phase 9+)

---

## Support

For deployment issues:
- **Streamlit Cloud:** [docs.streamlit.io](https://docs.streamlit.io)
- **Hugging Face:** [huggingface.co/docs/hub](https://huggingface.co/docs/hub)
- **Oracle Cloud:** Previous Oracle Cloud deployment guide
