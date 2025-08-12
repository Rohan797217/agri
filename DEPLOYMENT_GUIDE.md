# 🚀 GitHub to Streamlit Cloud Deployment Guide

## Step-by-Step Deployment Process

### 1. Prepare Your Repository for GitHub

#### Create requirements.txt for Streamlit Cloud
Since Replit uses `pyproject.toml`, you'll need a `requirements.txt` for Streamlit Cloud:

```bash
# Create requirements.txt from your current environment
pip freeze > requirements.txt
```

Or create manually with these dependencies:
```
streamlit>=1.28.0
groq>=0.4.0
ddgs>=9.5.0
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
trafilatura>=1.6.0
python-docx>=0.8.11
PyPDF2>=3.0.1
scikit-learn>=1.3.0
faiss-cpu>=1.7.4
```

#### Essential Files Already Created:
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `README.md` - Project documentation
- ✅ `.gitignore` - Git ignore patterns
- ✅ `DEPLOYMENT_GUIDE.md` - This guide

### 2. Push to GitHub

#### Initialize Git Repository:
```bash
git init
git add .
git commit -m "Initial commit: AgriAI Advisor with AGNO system"
```

#### Create GitHub Repository:
1. Go to [github.com](https://github.com) and create new repository
2. Name it: `agriAI-advisor` 
3. Add description: "AI-Powered Agricultural Advisory Platform for Indian Farmers"
4. Choose Public (for free Streamlit deployment) or Private (requires Streamlit Pro)
5. Don't initialize with README (you already have one)

#### Push to GitHub:
```bash
git remote add origin https://github.com/yourusername/agriAI-advisor.git
git branch -M main
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

#### Access Streamlit Cloud:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub account
3. Authorize Streamlit to access repositories

#### Create New App:
1. Click **"New app"**
2. **Repository**: Select `yourusername/agriAI-advisor`
3. **Branch**: `main` 
4. **Main file path**: `main.py`
5. **App URL**: Choose custom URL like `agriAI-advisor-yourname`

#### Configure Secrets:
1. Click **"Advanced settings"** 
2. In **"Secrets"** section, add:
```toml
GROQ_API_KEY = "your_actual_groq_api_key_here"
WEATHER_API_KEY = "your_weather_api_key_if_you_have_one"
```

#### Deploy:
1. Click **"Deploy!"**
2. Wait 3-5 minutes for deployment
3. Your app will be live at: `https://agriAI-advisor-yourname.streamlit.app`

### 4. Verify Deployment

#### Check App Status:
- Green indicator: ✅ App running successfully
- Red indicator: ❌ Check logs for errors
- Yellow indicator: ⚠️ App building/restarting

#### Common Issues & Solutions:

**Requirements Error:**
```
ERROR: No matching distribution found for package-name
```
**Solution**: Update requirements.txt with correct package versions

**Secrets Error:**
```
KeyError: 'GROQ_API_KEY'
```
**Solution**: Add API keys in Streamlit Cloud secrets section

**Import Error:**
```
ModuleNotFoundError: No module named 'agents'
```
**Solution**: Ensure all folders have `__init__.py` files

### 5. Continuous Deployment

#### Auto-Deploy on Git Push:
- Any push to `main` branch automatically redeploys
- Check deployment status in Streamlit Cloud dashboard
- View deployment logs for debugging

#### Manual Redeployment:
1. Go to Streamlit Cloud dashboard
2. Click your app
3. Click **"Reboot app"** if needed

### 6. Domain & Sharing

#### Share Your App:
- Direct URL: `https://agriAI-advisor-yourname.streamlit.app`
- Share on social media, embed in websites
- QR code generation for mobile access

#### Custom Domain (Pro Feature):
- Available with Streamlit Cloud Pro subscription
- Configure custom domain like `agriAI.yourdomain.com`

### 7. Monitoring & Maintenance

#### View Analytics:
- Streamlit Cloud provides basic usage analytics
- Monitor app performance and user engagement

#### Update Process:
1. Make changes locally
2. Test thoroughly
3. Push to GitHub: `git push origin main`
4. Automatic deployment in 2-3 minutes

---

## 🌾 Cotton Irrigation Schedule for Maharashtra

### Current Irrigation Advice for Your 0.20-acre Cotton Farm in Wardha:

#### **Immediate Actions (Next 7 Days):**

1. **Pre-Irrigation Soil Check:**
   - Insert finger 3-4 inches into soil
   - If dry at 3 inches depth → Irrigate within 24 hours
   - If moist → Wait 2-3 days and recheck

2. **Weather-Based Decision:**
   - **If no rain predicted for 5+ days:** Irrigate immediately
   - **If rain expected in 2-3 days:** Delay irrigation
   - **Check daily weather forecast:** Use IMD or AccuWeather

#### **Cotton Growth Stage Considerations:**

**Flowering/Boll Formation Stage (Current - October):**
- **Critical irrigation period** - Ensure adequate moisture
- **Frequency:** Every 12-15 days if no rainfall
- **Water requirement:** 15-20 liters per plant
- **Total for 0.20 acres:** Approximately 8,000-10,000 liters

#### **Irrigation Schedule:**

**Method 1: Basin Irrigation (Traditional)**
- **Timing:** Early morning (5-7 AM) or evening (5-7 PM)
- **Depth:** 6-8 cm standing water
- **Duration:** Until water penetrates 60-75 cm deep

**Method 2: Drip Irrigation (Recommended)**
- **Daily:** 2-3 hours in morning
- **Flow rate:** 2-4 liters/hour per emitter
- **Government subsidy:** 90% under PMKSY scheme

#### **Weather Monitoring Indicators:**

**Irrigate When:**
- Soil feels dry at 4-inch depth
- No rainfall for 7 days
- Temperature above 35°C for 3+ consecutive days
- Leaves show slight wilting in afternoon

**Delay Irrigation When:**
- Rain forecasted within 48 hours
- Soil moist at surface level
- Recent heavy rainfall (>25mm)

#### **Cost Estimation for Wardha:**

**Basin Irrigation:**
- **Electricity/Diesel:** ₹200-300 per irrigation
- **Labor:** ₹150-200 per day
- **Total per irrigation:** ₹350-500

**Drip System Installation:**
- **Initial cost:** ₹15,000-20,000 for 0.20 acres
- **Government subsidy:** ₹13,500-18,000 (90%)
- **Your cost:** ₹1,500-2,000
- **Monthly operation:** ₹300-500

#### **Emergency Stress Indicators:**

**Immediate irrigation needed if:**
- Leaves curl during morning hours
- Upper leaves become dull green
- Flowers start dropping excessively
- Boll development slows down

#### **Seasonal Calendar for Next 3 Months:**

**October 2025:**
- Irrigate every 10-12 days
- Focus on boll filling stage
- Monitor for bollworm activity

**November 2025:**
- Reduce to every 15-18 days
- Prepare for harvest season
- Last irrigation 15 days before picking

**December 2025:**
- Stop irrigation 2 weeks before harvest
- Allow natural field drying
- Begin harvest preparation

Would you like specific guidance on drip irrigation subsidies or help setting up weather monitoring for your farm?