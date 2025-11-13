# Streamlit Cloud Quick Start (5 Minutes)

## What You Get
- ✅ **Private dashboard** with email authentication
- ✅ **Free hosting** (up to 3 private apps)
- ✅ **Automatic HTTPS**
- ✅ **Auto-deploy** from GitHub (push code → app updates)
- ✅ **No server management**

## Prerequisites
- ✅ GitHub account
- ✅ Code already working locally (you have this!)
- ✅ 5 minutes of time

---

## Step-by-Step Deployment

### Step 1: Push Code to GitHub (2 minutes)

**If you haven't pushed to GitHub yet:**

```bash
cd c:/repos/bess-dashboard/bess-dashboard-1

# Check Git status
git status

# Add all files
git add .

# Commit
git commit -m "Deploy BESS Dashboard to Streamlit Cloud"

# Push to GitHub
git push origin main
```

**If you don't have a GitHub remote set up:**

1. Go to [github.com](https://github.com) and create new repository: `bess-dashboard`
2. Make it **public** (required for Streamlit Cloud free tier)
3. Add remote and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/bess-dashboard.git
git branch -M main
git push -u origin main
```

---

### Step 2: Deploy to Streamlit Cloud (3 minutes)

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**
   - Click "Continue with GitHub"
   - Authorize Streamlit Cloud to access your repos

3. **Click "New app" (top right)**

4. **Fill in deployment settings:**

   | Field | Value |
   |-------|-------|
   | **Repository** | `YOUR_USERNAME/bess-dashboard` |
   | **Branch** | `main` |
   | **Main file path** | `app.py` |
   | **App URL (optional)** | `bess-dashboard-yourname` |

5. **Click "Advanced settings"** (optional but recommended):

   - **Python version:** 3.10
   - **Make this a private app:** ✅ **CHECK THIS BOX**
     - This enables email authentication
     - Only people you authorize can access the dashboard

6. **Click "Deploy!"**

---

### Step 3: Watch Deployment (1 minute)

You'll see build logs in real-time:

```
[2025-11-13 10:15:32] Cloning repository...
[2025-11-13 10:15:35] Installing dependencies from requirements.txt...
[2025-11-13 10:16:45] Starting Streamlit app...
[2025-11-13 10:16:50] Your app is live! 🎉
```

**Your dashboard URL:** `https://bess-dashboard-yourname.streamlit.app`

---

### Step 4: Configure Access Control (Private Apps Only)

**If you enabled "Make this a private app":**

1. **Go to your app settings:**
   - Click the hamburger menu (☰) on your deployed app
   - Select "Settings"

2. **Navigate to "Sharing" tab**

3. **Add authorized viewers:**
   - Enter email addresses of people who should have access:
     - Finance team members
     - O&M team members
     - Your stakeholders

4. **How it works:**
   - User visits your app URL
   - Prompted to enter their email
   - Receives magic link to email
   - Clicks link → gets access (no password needed)
   - ⚠️ Only emails you added can access

---

### Step 5: Test Your Dashboard

1. **Open app URL** in incognito/private browsing mode:
   ```
   https://bess-dashboard-yourname.streamlit.app
   ```

2. **You should see:**
   - Email authentication screen (if private app)
   - Dashboard landing page after authentication

3. **Test full workflow:**
   - Go to "Data Upload" page
   - Upload SCADA CSV: `timestamp, power_mw, soc_percent`
   - Upload Market CSV: `timestamp, price_gbp_mwh, market_type`
   - Click "Process Data"
   - Go to "Optimization" page
   - Run optimization
   - View results on Finance/O&M/Optimization pages

---

## Updating Your App

**The beauty of Streamlit Cloud:** Just push to GitHub!

```bash
# Make changes to your code locally
# Edit files in VS Code, test locally with: streamlit run app.py

# Commit and push
git add .
git commit -m "Add new feature X"
git push origin main

# Streamlit Cloud automatically detects the push and redeploys (1-2 minutes)
```

**No manual deployment needed!** The app updates automatically.

---

## Troubleshooting

### Issue 1: "App is in an error state"

**Symptoms:** Red error banner on deployed app

**Solutions:**

1. **Check logs:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "Manage app"
   - Click "Logs" to see error messages

2. **Common causes:**
   - Missing dependency in `requirements.txt`
   - File path issues (use relative paths, not absolute)
   - Large data files committed to Git (should be in `.gitignore`)

3. **Fix:**
   ```bash
   # Fix the issue locally, then push
   git add .
   git commit -m "Fix deployment error"
   git push origin main
   ```

---

### Issue 2: "Out of memory"

**Symptoms:** App crashes when processing large CSV files

**Solutions:**

1. **Check file sizes:**
   - Free tier has 1 GB RAM limit
   - Recommended: Process files < 50 MB

2. **Optimize data processing:**
   - Process data in chunks
   - Clear intermediate variables with `del`
   - Use `st.cache_data()` efficiently

3. **Upgrade if needed:**
   - Streamlit Cloud paid tier: $20/month (more RAM)
   - Or switch to Oracle Cloud (free tier, 1 GB RAM, no limits)

---

### Issue 3: "App goes to sleep"

**Symptoms:** App takes 30 seconds to load after inactivity

**Cause:** Free tier apps sleep after 7 days of no usage

**Solutions:**

1. **Accept the behavior** (30-second wake time is acceptable for demos)
2. **Upgrade to paid tier** (apps never sleep)
3. **Use Oracle Cloud** for always-on deployment

---

### Issue 4: "Cannot find module X"

**Error:** `ModuleNotFoundError: No module named 'plotly'`

**Cause:** Missing dependency in `requirements.txt`

**Fix:**

1. Check your local `requirements.txt`:
   ```bash
   cat requirements.txt
   ```

2. Ensure all imports are listed:
   ```txt
   streamlit>=1.28.0
   pandas>=2.0.0
   plotly>=5.17.0
   PuLP>=2.7.0
   pydantic>=2.5.0
   PyYAML>=6.0
   ```

3. Push updated requirements:
   ```bash
   git add requirements.txt
   git commit -m "Fix dependencies"
   git push origin main
   ```

---

### Issue 5: "File not found: data/raw/..."

**Cause:** Absolute file paths used instead of relative paths

**Fix:**

In your code, use:
```python
# CORRECT (relative path)
Path("data/raw/scada.csv")

# WRONG (absolute path - won't work in cloud)
Path("C:/repos/bess-dashboard/data/raw/scada.csv")
```

---

## Managing Multiple Apps

**Free tier limit:** 3 private apps

**Strategy:**

1. **BESS Dashboard** (this project) - Private app #1
2. **Reserve slots** for future projects
3. **Public apps:** Unlimited (for portfolio/demos)

**If you need more private apps:**
- Upgrade to Streamlit Cloud Pro: $20/month (5 private apps)

---

## Monitoring & Analytics

### View App Metrics

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Manage app"
3. View:
   - **Active users** (real-time)
   - **Total sessions** (historical)
   - **Resource usage** (RAM, CPU)

### View Logs

```
Click "Logs" to see:
- Streamlit server logs
- Python print() statements
- Error tracebacks
- st.write() outputs
```

---

## Security Best Practices

### ✅ DO:
- Enable "Make this a private app" for commercial data
- Add only authorized emails to viewer list
- Use `.gitignore` to exclude data files
- Store API keys in Streamlit Secrets (not in code)

### ❌ DON'T:
- Commit `.env` files to Git
- Commit large CSV files (use `.gitignore`)
- Share your app URL publicly (if private app)
- Hardcode sensitive data in code

### Adding Secrets (API Keys, Passwords)

If you need to store secrets (e.g., database passwords):

1. **In Streamlit Cloud:**
   - Go to app Settings → Secrets
   - Add in TOML format:
     ```toml
     [database]
     host = "mydb.example.com"
     password = "secret123"
     ```

2. **In your code:**
   ```python
   import streamlit as st

   # Access secrets
   db_password = st.secrets["database"]["password"]
   ```

---

## Next Steps After Deployment

### Share with Stakeholders

**Email template:**

```
Subject: BESS Dashboard - Private Access Link

Hi [Stakeholder Name],

I've deployed the BESS Dashboard for your review:

🔗 URL: https://bess-dashboard-yourname.streamlit.app

Access Instructions:
1. Click the link above
2. Enter your email: [their_email@company.com]
3. Check your inbox for magic link
4. Click magic link to access dashboard

Features:
- Upload SCADA and Market Price CSV files
- Automated data quality analysis
- MILP optimization for arbitrage revenue
- Finance KPIs (Market Capture, Revenue Variance)
- O&M KPIs (Availability, Cycle Utilization)

Please test and provide feedback!

Best regards,
[Your Name]
```

### Collect Feedback

**Key questions to ask:**
- Is the dashboard intuitive to use?
- Are the KPIs relevant to your role (Finance vs O&M)?
- What additional features would be helpful?
- Any performance issues or bugs?

### Iterate

Based on feedback:
```bash
# Make improvements locally
# Test: streamlit run app.py

# Push updates
git add .
git commit -m "Add feature requested by stakeholders"
git push origin main

# App auto-updates in 1-2 minutes!
```

---

## Cost Comparison

| Tier | Cost | Features |
|------|------|----------|
| **Free** | $0/month | 3 private apps, unlimited public, 1 GB RAM |
| **Pro** | $20/month | 5 private apps, 3 GB RAM, priority support |
| **Team** | $250/month | 10 users, 6 GB RAM, advanced security |

**For your MVP:** Free tier is sufficient!

---

## Alternative: Oracle Cloud (If You Need More Resources)

If you encounter limitations on Streamlit Cloud:

- **Oracle Cloud Free Tier:** 1 GB RAM, always-on, no sleep
- **Deployment time:** 30 minutes
- **See:** `docs/DEPLOYMENT_GUIDE.md` (Oracle Cloud section)

---

## Support Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)
- **Status Page:** [status.streamlit.io](https://status.streamlit.io)

---

## Summary

**You just deployed a production-ready BESS Dashboard in 5 minutes!**

✅ **Private access control** (email authentication)
✅ **Auto-deploy from Git** (push code → app updates)
✅ **Zero server management** (Streamlit handles infrastructure)
✅ **Free hosting** (3 private apps on free tier)

**Your app URL:** `https://bess-dashboard-yourname.streamlit.app`

**Next:** Share with Finance and O&M stakeholders, collect feedback, iterate!
