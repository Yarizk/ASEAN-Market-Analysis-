from flask import Flask, render_template, send_from_directory
import os
import subprocess
from datetime import datetime

app = Flask(__name__)

directories = [
    'charts/correlations', 'charts/currency_analysis', 
    'charts/global_comparison', 'charts/market_performance', 
    'charts/risk_metrics', 'data', 'fred', 'reports', 
    'static/charts', 'static/data', 'templates'
]
for dir_name in directories:
    os.makedirs(dir_name, exist_ok=True)

def generate_analysis():
    try:
        subprocess.run(['python', 'fred.py'], check=True)
        subprocess.run(['python', 'YFI.py'], check=True)
        return True
    except Exception as e:
        print(f"Error generating analysis: {e}")
        return False

@app.route('/')
def index():
    success = generate_analysis()
    if not success:
        return "Error generating analysis. Check console for details."

    nav = {
        'Charts': {
            'Correlations': [
                'correlations/Currency_correlation_heatmap.png',
                'correlations/ETF_correlation_heatmap.png',
                'correlations/Index_correlation_heatmap.png'
            ],
            'Currency Analysis': [
                'currency_analysis/currency_performance.html',
                'currency_analysis/currency_volatility.html'
            ],
            'Global Comparison': [
                'global_comparison/asean_vs_major.html',
                'global_comparison/global_performance.html'
            ],
            'Market Performance': [
                'market_performance/index_performance.html',
                'market_performance/volume_analysis.html'
            ],
            'Risk Metrics': [
                'risk_metrics/risk_metrics_comparison.html',
                'risk_metrics/risk_profile_radar.html',
                'risk_metrics/risk_return_scatter.html'
            ]
        },
        'Data': [
            'correlations_Currency.csv',
            'correlations_ETF.csv',
            'correlations_Index.csv',
            'market_data.csv',
            'market_metrics.csv',
            'market_rankings.csv',
            'risk_metrics.csv'
            'fred_data.csv',
            'correlations.csv'
        ],
        'fred': [
            'correlation_heatmap.html',
            'macro_dashboard.html',
            'market_performance.html',
            'risk_return_scatter.html'
        ],
        'Reports': [
            'analysis_report.html'
        ]
    }
    
    return render_template('index.html', nav=nav, last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/charts/<path:path>')
def send_charts(path):
    return send_from_directory('charts', path)

@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)

@app.route('/fred/<path:path>')
def send_fred(path):
    return send_from_directory('fred', path)

@app.route('/reports/<path:path>')
def send_reports(path):
    return send_from_directory('reports', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)