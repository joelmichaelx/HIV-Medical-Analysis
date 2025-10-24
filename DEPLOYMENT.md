# HIV/AIDS Analytics Platform - Deployment Guide

This guide covers multiple deployment options for the HIV/AIDS Data Analysis Platform.

---

##  Recommended: Streamlit Community Cloud (FREE)

### Benefits
-  **100% Free** hosting
-  Built specifically for Streamlit apps
-  Deploy directly from GitHub
-  Automatic updates on git push
-  Easy setup (< 5 minutes)
-  Custom domain support

### Deployment Steps

#### 1. Push to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: HIV Analytics Platform"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/HIV-Medical-Analysis.git
git branch -M main
git push -u origin main
```

#### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/HIV-Medical-Analysis`
5. Set main file path: `src/visualization/dashboards/main_dashboard.py`
6. Click "Deploy"

**That's it!** Your app will be live in 2-3 minutes at:
`https://YOUR_USERNAME-hiv-medical-analysis.streamlit.app`

---

##  Alternative Deployment Options

### Option 2: Render (Free Tier Available)

#### Setup

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: hiv-analytics
    env: python
    buildCommand: pip install -r requirements_deploy.txt
    startCommand: streamlit run src/visualization/dashboards/main_dashboard.py --server.port $PORT --server.address 0.0.0.0
    plan: free
```

2. Push to GitHub
3. Go to [render.com](https://render.com)
4. Connect your GitHub repo
5. Deploy

**URL:** `https://hiv-analytics.onrender.com`

---

### Option 3: Railway (Free $5/month credit)

#### Setup

1. Install Railway CLI:

```bash
npm install -g @railway/cli
```

2. Login and deploy:

```bash
railway login
railway init
railway up
```

3. Set start command in Railway dashboard:

```bash
streamlit run src/visualization/dashboards/main_dashboard.py --server.port $PORT --server.address 0.0.0.0
```

---

### Option 4: Heroku (Paid)

#### Setup

1. Create `Procfile`:

```
web: sh setup.sh && streamlit run src/visualization/dashboards/main_dashboard.py
```

2. Create `setup.sh`:

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:

```bash
heroku create your-app-name
git push heroku main
heroku open
```

---

### Option 5: Docker + Cloud Run / AWS / Azure

#### Using the provided `docker-compose.yml`:

1. Build image:

```bash
docker build -t hiv-analytics .
```

2. Deploy to:
   - **Google Cloud Run**: `gcloud run deploy`
   - **AWS ECS**: Use AWS Console or CLI
   - **Azure Container Instances**: `az container create`

---

##  Environment Variables

For production deployment, set these environment variables:

```bash
# Optional: Database connections (if using real data)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Optional: API keys for external data sources
WHO_API_KEY=your_key_here
CDC_API_KEY=your_key_here
```

---

##  Security Considerations

### Before Deploying:

1. **Remove sensitive data** from config files
2. **Use environment variables** for secrets
3. **Enable authentication** if handling real patient data
4. **Set up HTTPS** (automatic on most platforms)
5. **Review .gitignore** to ensure no secrets are committed

### Important Files to Keep Private:

```
.env
config/*_prod.yaml
data/raw/*
*.db
*.sqlite
```

---

##  Performance Optimization

### For Production:

1. **Data Caching**: Already implemented with `@st.cache_data`
2. **Model Caching**: Already implemented with `@st.cache_resource`
3. **Limit Data Size**: Use sampling for large datasets
4. **CDN for Static Assets**: Use Streamlit Cloud's built-in CDN

### Recommended Settings:

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

---

##  Testing Before Deployment

Run these checks before deploying:

```bash
# 1. Test locally
streamlit run src/visualization/dashboards/main_dashboard.py

# 2. Check dependencies
pip list

# 3. Run tests (if available)
pytest tests/

# 4. Check for security issues
pip check
```

---

##  Monitoring & Logs

### Streamlit Cloud:

- View logs in the Streamlit Cloud dashboard
- Monitor app health and usage statistics
- Set up email notifications for errors

### Other Platforms:

- **Render**: Built-in logs dashboard
- **Railway**: Real-time logs in CLI or dashboard
- **Heroku**: `heroku logs --tail`

---

##  Continuous Deployment

### Automatic Deployments:

All recommended platforms support automatic deployments:

1. **Push to GitHub** → Automatically deploys
2. **No manual steps** required
3. **Rollback** available if issues occur

---

##  Cost Comparison

| Platform | Free Tier | Paid Starting | Best For |
|----------|-----------|---------------|----------|
| **Streamlit Cloud** |  Yes (3 apps) | $0 | Streamlit apps |
| **Render** |  Yes | $7/month | General apps |
| **Railway** | $5 credit/month | $0.000231/GB-hr | Scalable apps |
| **Heroku** |  No | $7/month | Enterprise |
| **Vercel** |  Not suitable | N/A | Static sites only |

---

##  Recommended Deployment Path

### For This Project:

** Best Choice: Streamlit Community Cloud**

**Why:**
1. Free forever (3 apps)
2. No credit card required
3. Designed for Streamlit
4. Zero configuration
5. Automatic HTTPS
6. Custom domains supported
7. Built-in secrets management

### Quick Start Command:

```bash
# 1. Create GitHub repo
git add .
git commit -m "Deploy HIV Analytics Platform"
git remote add origin https://github.com/YOUR_USERNAME/HIV-Medical-Analysis.git
git push -u origin main

# 2. Go to share.streamlit.io and deploy!
```

---

## 🆘 Troubleshooting

### Common Issues:

**Issue:** Dependencies fail to install
**Solution:** Use `requirements_deploy.txt` instead of `requirements.txt`

**Issue:** App crashes on startup
**Solution:** Check logs for missing modules, add to requirements

**Issue:** Slow loading
**Solution:** Reduce default data size, add more caching

**Issue:** Memory errors
**Solution:** Reduce `n_patients` slider range or use data sampling

---

##  Additional Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Render Deployment Guide](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app/)
- [Heroku Streamlit Buildpack](https://github.com/heroku/heroku-buildpack-python)

---

##  Pre-Deployment Checklist

- [ ] Git repository initialized
- [ ] All files committed
- [ ] Pushed to GitHub
- [ ] `requirements_deploy.txt` verified
- [ ] `.streamlit/config.toml` configured
- [ ] Sensitive data removed
- [ ] `.gitignore` updated
- [ ] README updated with deployment URL
- [ ] Local testing completed
- [ ] Choose deployment platform
- [ ] Deploy and test live URL

---

**Ready to deploy? Follow the Streamlit Cloud steps above for the easiest deployment!**

