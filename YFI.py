import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class GlobalMarketAnalysis:
    def __init__(self):
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=10*365)
        
        self.setup_folders()
        
        self.asean_indices = { 
            'Indonesia': '^JKSE',
            'Thailand': '^SET.BK',
            'Malaysia': '^KLSE',
            'Philippines': 'PSEI.PS',
            'Singapore': '^STI',
            # 'Vietnam': '^VNINDEX.VN'
        }
        
        self.global_indices = {
            'S&P500': '^GSPC',       # USA
            'NASDAQ': '^IXIC',       # USA Tech
            'DOW': '^DJI',           # USA
            'FTSE': '^FTSE',         # UK
            'DAX': '^GDAXI',         # Germany
            'CAC40': '^FCHI',        # France
            'NIKKEI': '^N225',       # Japan
            'HSI': '^HSI',           # Hong Kong
            'CSI300': '000300.SS',   # China
            'KOSPI': '^KS11',        # South Korea
            'ASX200': '^AXJO',       # Australia
            'SENSEX': '^BSESN'       # India
        }
        
        self.etfs = {
            'Indonesia': 'EIDO',
            'Thailand': 'THD',
            'Malaysia': 'EWM',
            'Philippines': 'EPHE',
            'Singapore': 'EWS',
            'Vietnam': 'VNM'
        }

        self.currencies = {
            'Indonesia': 'USDIDR=X',
            'Thailand': 'USDTHB=X',
            'Malaysia': 'USDMYR=X',
            'Philippines': 'USDPHP=X',
            'Singapore': 'USDSGD=X',
            'Vietnam': 'USDVND=X',
            'Euro': 'EURUSD=X',
            'GBP': 'GBPUSD=X',
            'JPY': 'USDJPY=X',
            'CNY': 'USDCNY=X'
        }
        
        plt.style.use('seaborn')
        self.colors = px.colors.qualitative.Set3

    def setup_folders(self):
        folders = [
            'data',
            'charts/market_performance',
            'charts/correlations',
            'charts/risk_metrics',
            'charts/currency_analysis',
            'charts/global_comparison',
            'reports'
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def fetch_market_data(self):
        all_data = []
        
        print("\nFetching ASEAN market data...")
        for country, ticker in self.asean_indices.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    df = df[['Adj Close', 'Volume']].copy()
                    df.columns = [f'ASEAN_{country}_Index', f'ASEAN_{country}_Volume']
                    all_data.append(df)
                    print(f"Successfully fetched {country} data")
            except Exception as e:
                print(f"Error fetching {country} data: {e}")

        print("\nFetching global market data...")
        for market, ticker in self.global_indices.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    df = df[['Adj Close', 'Volume']].copy()
                    df.columns = [f'Global_{market}_Index', f'Global_{market}_Volume']
                    all_data.append(df)
                    print(f"Successfully fetched {market} data")
            except Exception as e:
                print(f"Error fetching {market} data: {e}")

        print("\nFetching ETF data...")
        for country, ticker in self.etfs.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    df = df[['Adj Close', 'Volume']].copy()
                    df.columns = [f'ETF_{country}', f'ETF_{country}_Volume']
                    all_data.append(df)
                    print(f"Successfully fetched {country} ETF data")
            except Exception as e:
                print(f"Error fetching {country} ETF: {e}")

        # Fetch currencies
        print("\nFetching currency data...")
        for country, ticker in self.currencies.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    df = df[['Adj Close']].copy()
                    df.columns = [f'Currency_{country}']
                    all_data.append(df)
                    print(f"Successfully fetched {country} currency data")
            except Exception as e:
                print(f"Error fetching {country} currency: {e}")


        if all_data:
            combined_data = pd.concat(all_data, axis=1)
            combined_data = combined_data.fillna(method='ffill')
            return combined_data
        else:
            raise ValueError("No data was successfully fetched")

    def create_global_comparison_charts(self, df):
        print("\nGenerating global market comparison charts...")
        
        global_indices = [col for col in df.columns if 'Global_' in col and 'Index' in col]
        fig = go.Figure()
        
        for col in global_indices:
            normalized_price = df[col] / df[col].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=col.replace('Global_', '').replace('_Index', ''),
                mode='lines'
            ))
            
        fig.update_layout(
            title='Global Market Performance (Normalized)',
            xaxis_title='Date',
            yaxis_title='Index (Base=100)',
            template='plotly_white',
            height=600,
            width=1000
        )
        fig.write_html('charts/global_comparison/global_performance.html')
        
        asean_indices = [col for col in df.columns if 'ASEAN_' in col and 'Index' in col]
        major_indices = ['Global_S&P500_Index', 'Global_NIKKEI_Index', 'Global_HSI_Index']
        
        fig = go.Figure()
        
        for col in asean_indices:
            normalized_price = df[col] / df[col].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=col.replace('ASEAN_', '').replace('_Index', ''),
                mode='lines',
                line=dict(dash='solid')
            ))
            
        for col in major_indices:
            normalized_price = df[col] / df[col].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=col.replace('Global_', '').replace('_Index', ''),
                mode='lines',
                line=dict(dash='dash')
            ))
            
        fig.update_layout(
            title='ASEAN vs Major Markets Performance',
            xaxis_title='Date',
            yaxis_title='Index (Base=100)',
            template='plotly_white',
            height=600,
            width=1000
        )
        fig.write_html('charts/global_comparison/asean_vs_major.html')
    def calculate_metrics(self, df):
        metrics = pd.DataFrame()
        
        all_indices = {**self.asean_indices, **self.global_indices}

        for country, _ in all_indices.items():
            try:
                prefix = 'ASEAN_' if country in self.asean_indices else 'Global_'
                
                index_returns = df[f'{prefix}{country}_Index'].pct_change()
                
                metrics.loc[country, 'Annual_Return'] = index_returns.mean() * 252
                metrics.loc[country, 'Annual_Volatility'] = index_returns.std() * np.sqrt(252)
                metrics.loc[country, 'Sharpe_Ratio'] = metrics.loc[country, 'Annual_Return'] / metrics.loc[country, 'Annual_Volatility']
                
                currency_key = country if country in self.currencies else None
                if currency_key and f'Currency_{currency_key}' in df.columns:
                    currency_returns = df[f'Currency_{currency_key}'].pct_change()
                    metrics.loc[country, 'Currency_Volatility'] = currency_returns.std() * np.sqrt(252)
                
                if f'{prefix}{country}_Volume' in df.columns:
                    recent_volume = df[f'{prefix}{country}_Volume'].tail(60).mean()
                    old_volume = df[f'{prefix}{country}_Volume'].head(60).mean()
                    metrics.loc[country, 'Volume_Growth'] = (recent_volume / old_volume - 1) * 100
                
                if country in self.etfs and f'ETF_{country}' in df.columns:
                    etf_returns = df[f'ETF_{country}'].pct_change()
                    metrics.loc[country, 'ETF_Annual_Return'] = etf_returns.mean() * 252
                    metrics.loc[country, 'ETF_Tracking_Error'] = (index_returns - etf_returns).std() * np.sqrt(252)
                
                cum_returns = (1 + index_returns).cumprod()
                rolling_max = cum_returns.cummax()
                drawdowns = (cum_returns - rolling_max) / rolling_max
                metrics.loc[country, 'Max_Drawdown'] = drawdowns.min()
                
                ihsg_returns = df['ASEAN_Indonesia_Index'].pct_change()
                metrics.loc[country, 'Beta_to_IHSG'] = index_returns.cov(ihsg_returns) / ihsg_returns.var()
                
            except Exception as e:
                print(f"Error calculating metrics for {country}: {e}")

        return metrics

    def calculate_correlations(self, df):
        correlations = {}
        
        index_cols = [col for col in df.columns if 'Index' in col and 'Volume' not in col]
        if index_cols:
            correlations['Index'] = df[index_cols].pct_change().corr()
        
        etf_cols = [col for col in df.columns if 'ETF' in col and 'Volume' not in col]
        if etf_cols:
            correlations['ETF'] = df[etf_cols].pct_change().corr()
        
        currency_cols = [col for col in df.columns if 'Currency' in col]
        if currency_cols:
            correlations['Currency'] = df[currency_cols].pct_change().corr()
        
        return correlations
    def setup_folders(self):
        folders = ['data', 'charts/market_performance', 
                  'charts/correlations', 'charts/risk_metrics', 
                  'charts/currency_analysis', 'reports']
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def create_market_performance_charts(self, df):
        print("Generating market performance charts...")
        
        all_indices = {**self.asean_indices, **self.global_indices}
        
        fig = go.Figure()
        for country in all_indices.keys():
            prefix = 'ASEAN_' if country in self.asean_indices else 'Global_'
            normalized_price = df[f'{prefix}{country}_Index'] / df[f'{prefix}{country}_Index'].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=country,
                mode='lines'
            ))
            
        fig.update_layout(
            title='Market Performance (Normalized)',
            xaxis_title='Date',
            yaxis_title='Index (Base=100)',
            template='plotly_white',
            height=600
        )
        fig.write_html('charts/market_performance/index_performance.html')
        
        fig = make_subplots(rows=len(all_indices), cols=1,
                        subplot_titles=list(all_indices.keys()))
        
        row = 1
        for country in all_indices.keys():
            prefix = 'ASEAN_' if country in self.asean_indices else 'Global_'
            if f'{prefix}{country}_Volume' in df.columns:
                fig.add_trace(
                    go.Bar(x=df.index, y=df[f'{prefix}{country}_Volume'],
                        name=country),
                    row=row, col=1
                )
                row += 1
                
        fig.update_layout(height=200*len(all_indices), showlegend=False,
                        title_text="Trading Volume Analysis")
        fig.write_html('charts/market_performance/volume_analysis.html')

    def create_correlation_heatmaps(self, correlations):
        print("Generating correlation heatmaps...")
        
        for key, corr_matrix in correlations.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', center=0)
            plt.title(f'{key} Correlations')
            plt.tight_layout()
            plt.savefig(f'charts/correlations/{key}_correlation_heatmap.png')
            plt.close()

    def create_risk_metrics_charts(self, metrics):
        print("Generating risk metrics charts...")
        
        size_values = (metrics['Sharpe_Ratio'] - metrics['Sharpe_Ratio'].min() + 1) * 20
        
        fig = px.scatter(metrics, 
                        x='Annual_Volatility', 
                        y='Annual_Return',
                        text=metrics.index,
                        size=size_values,  
                        title='Risk-Return Analysis',
                        labels={
                            'Annual_Volatility': 'Annual Volatility',
                            'Annual_Return': 'Annual Return'
                        })
        
        fig.update_traces(
            hovertemplate="<br>".join([
                "Country: %{text}",
                "Return: %{y:.2%}",
                "Volatility: %{x:.2%}",
                f"Sharpe Ratio: {metrics['Sharpe_Ratio'].round(2)}"
            ])
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=600,
            width=800,
            showlegend=False,
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%'
        )
        fig.write_html('charts/risk_metrics/risk_return_scatter.html')
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Annual Returns', 'Annual Volatility',
                                         'Sharpe Ratio', 'Maximum Drawdown'))
        
        fig.add_trace(
            go.Bar(x=metrics.index, 
                  y=metrics['Annual_Return'],
                  text=metrics['Annual_Return'].apply(lambda x: f'{x:.1%}'),
                  textposition='auto',
                  name='Annual Return'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=metrics.index, 
                  y=metrics['Annual_Volatility'],
                  text=metrics['Annual_Volatility'].apply(lambda x: f'{x:.1%}'),
                  textposition='auto',
                  name='Volatility'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=metrics.index, 
                  y=metrics['Sharpe_Ratio'],
                  text=metrics['Sharpe_Ratio'].round(2),
                  textposition='auto',
                  name='Sharpe Ratio'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=metrics.index, 
                  y=metrics['Max_Drawdown'],
                  text=metrics['Max_Drawdown'].apply(lambda x: f'{x:.1%}'),
                  textposition='auto',
                  name='Max Drawdown'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            width=1000,
            showlegend=False,
            title_text="Risk Metrics Comparison"
        )
        
        fig.update_yaxes(tickformat='.1%', row=1, col=1)
        fig.update_yaxes(tickformat='.1%', row=1, col=2)
        fig.update_yaxes(tickformat='.1%', row=2, col=2)
        
        fig.write_html('charts/risk_metrics/risk_metrics_comparison.html')
        
        categories = ['Annual_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Annual_Volatility']
        
        normalized_metrics = pd.DataFrame()
        for cat in categories:
            if cat in ['Max_Drawdown', 'Annual_Volatility']:
                normalized_metrics[cat] = (metrics[cat] - metrics[cat].min()) / (metrics[cat].max() - metrics[cat].min())
                normalized_metrics[cat] = 1 - normalized_metrics[cat]
            else:
                normalized_metrics[cat] = (metrics[cat] - metrics[cat].min()) / (metrics[cat].max() - metrics[cat].min())
        
        fig = go.Figure()
        
        for country in normalized_metrics.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized_metrics.loc[country],
                theta=categories,
                fill='toself',
                name=country
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Risk Profile Comparison',
            height=600,
            width=800
        )
        
        fig.write_html('charts/risk_metrics/risk_profile_radar.html')
    def create_global_comparison_charts(self, df):
        print("\nGenerating global market comparison charts...")
        
        fig = go.Figure()
        for country, _ in self.global_indices.items():
            normalized_price = df[f'Global_{country}_Index'] / df[f'Global_{country}_Index'].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=country,
                mode='lines'
            ))
            
        fig.update_layout(
            title='Global Market Performance (Normalized)',
            xaxis_title='Date',
            yaxis_title='Index (Base=100)',
            template='plotly_white',
            height=600,
            width=1000
        )
        fig.write_html('charts/global_comparison/global_performance.html')
        
        asean_indices = [f'ASEAN_{country}_Index' for country in self.asean_indices.keys()]
        major_indices = ['Global_S&P500_Index', 'Global_NIKKEI_Index', 'Global_HSI_Index']
        
        fig = go.Figure()
        
        for col in asean_indices:
            normalized_price = df[col] / df[col].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=col.replace('ASEAN_', '').replace('_Index', ''),
                mode='lines',
                line=dict(dash='solid')
            ))
            
        for col in major_indices:
            normalized_price = df[col] / df[col].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_price,
                name=col.replace('Global_', '').replace('_Index', ''),
                mode='lines',
                line=dict(dash='dash')
            ))
            
        fig.update_layout(
            title='ASEAN vs Major Markets Performance',
            xaxis_title='Date',
            yaxis_title='Index (Base=100)',
            template='plotly_white',
            height=600,
            width=1000
        )
        fig.write_html('charts/global_comparison/asean_vs_major.html')
    def create_currency_analysis_charts(self, df):
        print("Generating currency analysis charts...")
        
        fig = go.Figure()
        for country, _ in self.currencies.items():
            if f'Currency_{country}' in df.columns:
                normalized_rate = df[f'Currency_{country}'] / df[f'Currency_{country}'].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized_rate,
                    name=country,
                    mode='lines'
                ))
                
        fig.update_layout(
            title='Currency Performance vs USD (Normalized)',
            xaxis_title='Date',
            yaxis_title='Exchange Rate (Base=100)',
            template='plotly_white'
        )
        fig.write_html('charts/currency_analysis/currency_performance.html')
        
        currency_volatility = pd.DataFrame()
        for country, _ in self.currencies.items():
            col_name = f'Currency_{country}'
            if col_name in df.columns:
                currency_volatility[country] = df[col_name].pct_change().rolling(30).std() * np.sqrt(252)
        
        fig = px.line(currency_volatility, title='30-Day Rolling Currency Volatility')
        fig.write_html('charts/currency_analysis/currency_volatility.html')
    def generate_html_report(self, metrics, correlations):
        print("Generating HTML report...")
        
        correlations_html = ""
        for key, corr_df in correlations.items():
            correlations_html += f"<h3>{key} Correlations</h3>"
            correlations_html += corr_df.to_html()
        
        html_content = f"""
        <html>
        <head>
            <title>Global Markets Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-card {{ 
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Global Markets Analysis Report</h1>
            
            <h2>Market Metrics</h2>
            {metrics.round(4).to_html()}
            
            <h2>Correlations</h2>
            {correlations_html}
            
            <h2>Key Findings</h2>
            <div class="metric-card">
                <h3>Best Performing Market</h3>
                <p>{metrics['Annual_Return'].idxmax()} with {metrics['Annual_Return'].max():.2%} annual return</p>
            </div>
            
            <div class="metric-card">
                <h3>Most Stable Market</h3>
                <p>{metrics['Annual_Volatility'].idxmin()} with {metrics['Annual_Volatility'].min():.2%} volatility</p>
            </div>
            
            <h2>Charts Location</h2>
            <ul>
                <li>Market Performance: charts/market_performance/</li>
                <li>Correlations: charts/correlations/</li>
                <li>Risk Metrics: charts/risk_metrics/</li>
                <li>Currency Analysis: charts/currency_analysis/</li>
            </ul>
        </body>
        </html>
        """
        
        with open('reports/analysis_report.html', 'w') as f:
            f.write(html_content)

    def generate_report(self):
        df = self.fetch_market_data()
        metrics = self.calculate_metrics(df)
        correlations = self.calculate_correlations(df)
        
        ranking_factors = ['Annual_Return', 'Sharpe_Ratio', 'Currency_Volatility', 'Max_Drawdown']
        rankings = pd.DataFrame()
        
        for factor in ranking_factors:
            if factor in metrics.columns:
                if factor in ['Currency_Volatility', 'Max_Drawdown']:
                    rankings[f'{factor}_Rank'] = metrics[factor].rank()
                else:
                    rankings[f'{factor}_Rank'] = metrics[factor].rank(ascending=False)
        
        rankings['Overall_Rank'] = rankings.mean(axis=1)
        rankings = rankings.sort_values('Overall_Rank')
        
        self.create_market_performance_charts(df)
        self.create_correlation_heatmaps(correlations)
        self.create_risk_metrics_charts(metrics)
        self.create_currency_analysis_charts(df)
        
        self.generate_html_report(metrics, rankings)
        
        df.to_csv('data/market_data.csv')
        metrics.to_csv('data/market_metrics.csv')
        rankings.to_csv('data/market_rankings.csv')
        
        for key, corr in correlations.items():
            corr.to_csv(f'data/correlations_{key}.csv')
        
        return df, metrics, correlations, rankings

    def generate_expanded_report(self):
        df = self.fetch_market_data()
        
        metrics = self.calculate_metrics(df)
        correlations = self.calculate_correlations(df)
        
        self.create_market_performance_charts(df)
        self.create_correlation_heatmaps(correlations)
        self.create_risk_metrics_charts(metrics)
        self.create_currency_analysis_charts(df)
        self.create_global_comparison_charts(df)
        
        self.generate_html_report(metrics, correlations)
        
        df.to_csv('data/market_data.csv')
        metrics.to_csv('data/market_metrics.csv')
        
        for key, corr in correlations.items():
            corr.to_csv(f'data/correlations_{key}.csv')
        
        return df, metrics, correlations

def main():
    analyzer = GlobalMarketAnalysis()
    
    try:
        print("Starting global market analysis...")
        df, metrics, correlations = analyzer.generate_expanded_report()
        print("\nAnalysis complete. Results saved in respective folders.")
        print("\nFolder structure:")
        print("- data/: Raw data and metrics")
        print("- charts/: Interactive visualizations")
        print("- reports/: HTML report")
        
        return df, metrics, correlations
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None, None, None

if __name__ == "__main__":
    df, metrics, correlations = main()