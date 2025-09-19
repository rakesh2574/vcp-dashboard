# VCP Dashboard - Automated Trading Pattern Analyzer

## 📈 Overview
This repository contains an automated VCP (Volatility Contraction Pattern) analyzer that:
- Automatically scans configured symbols for VCP patterns
- Runs twice daily via GitHub Actions (9:00 AM and 3:00 PM IST)
- Provides a web dashboard for viewing results

## 🚀 Live Dashboard
Access the viewer dashboard: 

## 📁 Repository Structure
```
vcp_dashboard/
├── .github/
│   └── workflows/
│       └── run_vcp_analyzer.yml    # Automated scheduler
├── vcp_classic_csv.py              # Main analyzer (runs headless)
├── vcp_user_dashboard.py           # User dashboard for viewing results
├── vcp_config.json                 # Configuration file
├── vcp_analysis_results.csv        # Latest analysis results
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Configuration
Edit `vcp_config.json` to customize:
- **symbols_input**: Comma-separated list of symbols to analyze
- **exchange**: NSE, BSE, BINANCE, NASDAQ, NYSE, or FX_IDC
- **timeframe**: 5m, 15m, 30m, 1H, 4H, or 1D
- **n_bars**: Number of historical bars (200-2000)
- **swing_length**: Swing high detection period (5-30)
- **zone_percent**: Resistance zone size (0.5-3.0%)
- **target_percent**: Profit target (5-30%)
- **stoploss_percent**: Stop loss (2-15%)

## 🤖 Automated Analysis Schedule
The analyzer runs automatically via GitHub Actions:
- **Morning Run**: 9:00 AM IST (Mon-Fri)
- **Afternoon Run**: 3:00 PM IST (Mon-Fri)
- **Manual Trigger**: Via GitHub Actions tab

## 📊 Pattern Types Detected
- **Setup Ready**: Valid VCP pattern ready for entry
- **Ongoing**: Active trades being monitored
- **Target Hit**: Successful trades that hit target
- **SL Hit**: Trades stopped out

## 🔑 Access Credentials
### Configuration Page (Admin)
- Username: `sherlock`
- Password: `irene`

### Viewer Dashboard (Users)
- Username: `sherlock`  
- Password: `watson`

## 💻 Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/vcp_dashboard.git
cd vcp_dashboard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run analyzer in headless mode
python vcp_classic_csv.py --headless

# Run viewer dashboard
streamlit run vcp_user_dashboard.py

# Run analyzer with UI (for configuration)
streamlit run vcp_classic_csv.py
```

## 📈 Latest Analysis Results
The most recent analysis results are automatically updated in `vcp_analysis_results.csv`

## ⚙️ Manual Trigger
To run the analyzer manually:
1. Go to the Actions tab in this repository
2. Select "Run VCP Analyzer"
3. Click "Run workflow"

## 📝 License
This project is for educational purposes. Use at your own risk.

## ⚠️ Disclaimer
This tool is for informational purposes only and should not be considered as financial advice. Always do your own research before making investment decisions.