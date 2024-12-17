import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from fredapi import Fred
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_white"
import warnings
warnings.filterwarnings('ignore')
class FREDMarketAnalysis:
    def __init__(self, fred_api_key):
        self.fred = Fred(api_key=fred_api_key)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=10*365)

        self.fred_indicators = {
            'US_CPI': 'CPIAUCSL',
            'FED_RATE': 'FEDFUNDS',
            'US_10Y': 'DGS10',
            'US_2Y': 'DGS2',
            'VIX': 'VIXCLS',
            'US_DOLLAR': 'DTWEXBGS',
            'US_M2': 'M2SL',
            'US_INDPRO': 'INDPRO',
            'US_UNRATE': 'UNRATE',
            'US_RETAIL': 'RSAFS'
        }
        
        self.market_tickers = {
            'Indonesia': {'index': '^JKSE', 'etf': 'EIDO'},
            'Singapore': {'index': '^STI', 'etf': 'EWS'},
            'Malaysia': {'index': '^KLSE', 'etf': 'EWM'},
            'Thailand': {'index': '^SET.BK', 'etf': 'THD'},
            'Philippines': {'index': 'PSEI.PS', 'etf': 'EPHE'}
        }

    def create_market_performance_chart(self, market_data):
        fig = go.Figure()
        
        for country in self.market_tickers.keys():
            if f'{country}_Index' in market_data.columns:
                normalized_data = market_data[f'{country}_Index'] / market_data[f'{country}_Index'].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=normalized_data.index,
                    y=normalized_data,
                    name=f'{country} Index',
                    mode='lines'
                ))
        
        fig.update_layout(
            title='ASEAN Markets Performance (Normalized to 100)',
            xaxis_title='Date',
            yaxis_title='Index Value',
            height=600,
            template='plotly_white'
        )
        
        return fig

    def create_correlation_heatmap(self, analysis):
        if 'correlations' in analysis:
            fig = px.imshow(
                analysis['correlations'],
                title='Macro Factor Correlations',
                color_continuous_scale='RdBu',
                aspect='auto',
                height=500
            )
            fig.update_layout(template='plotly_white')
            return fig
        return None

    def create_risk_return_scatter(self, analysis):
        if 'risk_metrics' in analysis:
            metrics = analysis['risk_metrics']
            fig = px.scatter(
                metrics,
                x='Volatility',
                y='Annual_Return',
                text=metrics.index,
                title='Risk-Return Analysis',
                height=500
            )
            
            fig.update_traces(
                textposition='top center',
                marker=dict(size=12)
            )
            
            fig.update_layout(
                template='plotly_white',
                xaxis_title='Annualized Volatility',
                yaxis_title='Annualized Return'
            )
            
            return fig
        return None

    def create_macro_dashboard(self, fred_data):

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Fed Funds Rate', 'US CPI', 'US 10Y Yield', 
                          'VIX Index', 'US Dollar Index', 'Industrial Production')
        )

        if 'FED_RATE' in fred_data.columns:
            fig.add_trace(
                go.Scatter(x=fred_data.index, y=fred_data['FED_RATE'],
                          name='Fed Rate', line=dict(color='red')),
                row=1, col=1
            )

        if 'US_CPI' in fred_data.columns:
            fig.add_trace(
                go.Scatter(x=fred_data.index, y=fred_data['US_CPI'].pct_change(12)*100,
                          name='CPI YoY%', line=dict(color='blue')),
                row=1, col=2
            )

        if 'US_10Y' in fred_data.columns:
            fig.add_trace(
                go.Scatter(x=fred_data.index, y=fred_data['US_10Y'],
                          name='10Y Yield', line=dict(color='green')),
                row=2, col=1
            )

        if 'VIX' in fred_data.columns:
            fig.add_trace(
                go.Scatter(x=fred_data.index, y=fred_data['VIX'],
                          name='VIX', line=dict(color='purple')),
                row=2, col=2
            )
        
        if 'US_DOLLAR' in fred_data.columns:
            fig.add_trace(
                go.Scatter(x=fred_data.index, y=fred_data['US_DOLLAR'],
                          name='USD Index', line=dict(color='orange')),
                row=3, col=1
            )
        
        if 'US_INDPRO' in fred_data.columns:
            fig.add_trace(
                go.Scatter(x=fred_data.index, y=fred_data['US_INDPRO'].pct_change(12)*100,
                          name='Industrial Production YoY%', line=dict(color='brown')),
                row=3, col=2
            )
        
        fig.update_layout(height=900, width=1000, showlegend=True,
                         title_text="US Macro Dashboard")
        
        return fig

    def generate_visual_report(self):
        fred_data = self.fetch_fred_data()
        market_data = self.fetch_market_data()
        analysis = self.analyze_data(fred_data, market_data)
        
        market_perf_chart = self.create_market_performance_chart(market_data)
        correlation_heatmap = self.create_correlation_heatmap(analysis)
        risk_return_scatter = self.create_risk_return_scatter(analysis)
        macro_dashboard = self.create_macro_dashboard(fred_data)
        
        market_perf_chart.write_html("fred/market_performance.html")
        correlation_heatmap.write_html("fred/correlation_heatmap.html")
        risk_return_scatter.write_html("fred/risk_return_scatter.html")
        macro_dashboard.write_html("fred/macro_dashboard.html")
        
        return fred_data, market_data, analysis

    def fetch_fred_data(self):
        fred_data = pd.DataFrame()
        
        for name, series_id in self.fred_indicators.items():
            try:
                data = self.fred.get_series(series_id, 
                                          observation_start=self.start_date,
                                          observation_end=self.end_date)
                fred_data[name] = data
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        return fred_data

    def fetch_market_data(self):
        market_data = pd.DataFrame()
        
        for country, tickers in self.market_tickers.items():
            try:
                index_data = yf.download(tickers['index'], 
                                       start=self.start_date,
                                       end=self.end_date,
                                       progress=False)
                index_data.index = index_data.index.tz_localize(None)
                market_data[f'{country}_Index'] = index_data['Adj Close']
                
                etf_data = yf.download(tickers['etf'],
                                     start=self.start_date,
                                     end=self.end_date,
                                     progress=False)
                etf_data.index = etf_data.index.tz_localize(None)
                market_data[f'{country}_ETF'] = etf_data['Adj Close']
                
            except Exception as e:
                print(f"Error fetching {country} data: {e}")
        
        return market_data

    def analyze_data(self, fred_data, market_data):
        analysis = {}
        
        fred_data = fred_data.resample('D').ffill()
        
        correlations = pd.DataFrame()
        
        for country in self.market_tickers.keys():
            if f'{country}_Index' in market_data.columns:
                market_returns = market_data[f'{country}_Index'].pct_change()
                
                for indicator in ['US_CPI', 'FED_RATE', 'US_10Y', 'VIX', 'US_DOLLAR']:
                    if indicator in fred_data.columns:
                        indicator_change = fred_data[indicator].pct_change()
                        aligned_data = pd.concat([market_returns, indicator_change], axis=1).dropna()
                        if not aligned_data.empty:
                            correlations.loc[country, f'{indicator}_Correlation'] = \
                                aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
        
        analysis['correlations'] = correlations
        
        risk_metrics = pd.DataFrame()
        
        for country in self.market_tickers.keys():
            if f'{country}_Index' in market_data.columns:
                returns = market_data[f'{country}_Index'].pct_change().dropna()
                
                risk_metrics.loc[country, 'Annual_Return'] = returns.mean() * 252
                risk_metrics.loc[country, 'Volatility'] = returns.std() * np.sqrt(252)
                risk_metrics.loc[country, 'Sharpe_Ratio'] = risk_metrics.loc[country, 'Annual_Return'] / \
                                                          risk_metrics.loc[country, 'Volatility']
                cum_returns = (1 + returns).cumprod()
                rolling_max = cum_returns.cummax()
                drawdowns = (cum_returns - rolling_max) / rolling_max
                risk_metrics.loc[country, 'Max_Drawdown'] = drawdowns.min()
        
        analysis['risk_metrics'] = risk_metrics
        
        return analysis

    def generate_report(self):
        fred_data = self.fetch_fred_data()
        market_data = self.fetch_market_data()
        
        analysis = self.analyze_data(fred_data, market_data)

        rankings = pd.DataFrame()
        
        if 'risk_metrics' in analysis:
            for col in analysis['risk_metrics'].columns:
                if col in ['Annual_Return', 'Sharpe_Ratio']:
                    rankings[f'{col}_Rank'] = analysis['risk_metrics'][col].rank(ascending=False)
                else:
                    rankings[f'{col}_Rank'] = analysis['risk_metrics'][col].rank()
        
        rankings['Overall_Rank'] = rankings.mean(axis=1)
        rankings = rankings.sort_values('Overall_Rank')

        report = "=== ASEAN Markets Macro Analysis Report ===\n\n"
        
        report += "1. Market Rankings:\n"
        report += str(rankings) + "\n\n"
        
        report += "2. Risk Metrics:\n"
        report += str(analysis['risk_metrics']) + "\n\n"
        
        report += "3. Macro Correlations:\n"
        report += str(analysis['correlations']) + "\n\n"
        
        return report, fred_data, market_data, analysis, rankings

def main():
    analyzer = FREDMarketAnalysis('get your own fred api key')
    fred_data, market_data, analysis = analyzer.generate_visual_report()

    fred_data.to_csv('data/fred_data.csv')
    market_data.to_csv('data/market_data.csv')
    
    for name, data in analysis.items():
        data.to_csv(f'data/{name}.csv')

if __name__ == "__main__":
    main()