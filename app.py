from flask import Flask, request, render_template, jsonify
import yfinance as yf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return [{'Date': index, 'Open': row['Open'], 'High': row['High'], 'Low': row['Low'], 'Close': row['Close']} for index, row in data.iterrows()]

def get_stock_data_visualization(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data.reset_index()

previous_audits = [
    {"id": 1, "date": "2024-05-10", "result": "Pass"},
    {"id": 2, "date": "2024-05-09", "result": "Fail"},
    {"id": 3, "date": "2024-05-08", "result": "Pass"},
 
]

def identify_trading_signals(ohlc_data):
    signals = []
    for i in range(2, len(ohlc_data)):
        # Check for Three White Soldiers pattern
        if (ohlc_data[i-2]['Close'] < ohlc_data[i-1]['Close'] < ohlc_data[i]['Close'] and
            ohlc_data[i-2]['Open'] < ohlc_data[i-1]['Open'] < ohlc_data[i]['Open']):
            signals.append({'pattern': 'Three White Soldiers', 'index': i})
        # Check for Three Black Crows pattern
        elif (ohlc_data[i-2]['Close'] > ohlc_data[i-1]['Close'] > ohlc_data[i]['Close'] and
              ohlc_data[i-2]['Open'] > ohlc_data[i-1]['Open'] > ohlc_data[i]['Open']):
            signals.append({'pattern': 'Three Black Crows', 'index': i})
    return signals

def calculate_var(data, confidence_level):
  
    sorted_data = np.sort(data)
    index = int((1 - confidence_level) * len(sorted_data))
    var = sorted_data[index]
    return var


def generate_chart(var_95, var_99):
    
    x = ['95%', '99%']
    y = [var_95, var_99]

    plt.bar(x, y)
    plt.xlabel('Confidence Level')
    plt.ylabel('VaR Value')
    plt.title('VaR Values for 95% and 99% Confidence Levels')
    plt.grid(True)
    chart_filename = 'var_chart.png'
    plt.savefig(chart_filename)
    
    plt.close()

    return chart_filename


def monte_carlo_analysis(price_data, num_simulations, confidence_levels):
    returns = [(price_data[i]['Close'] - price_data[i-1]['Close']) / price_data[i-1]['Close'] 
               for i in range(1, len(price_data))]
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    
    results = {'95%': [], '99%': []}
    
    for _ in range(num_simulations):
        simulated_returns = np.random.normal(mean_return, std_dev, len(returns))
        simulated_prices = [price_data[0]['Close']]
        for ret in simulated_returns:
            simulated_prices.append(simulated_prices[-1] * (1 + ret))
        
        simulated_changes = [(simulated_prices[i] - simulated_prices[0]) / simulated_prices[0] for i in range(1, len(simulated_prices))]
        sorted_changes = sorted(simulated_changes)
        
        for level in confidence_levels:
            index = int(level * len(sorted_changes))
            results[f'{int(level * 100)}%'].append(sorted_changes[index])
    
    return results


def assess_profitability(price_data, signals, num_days):
    results = []
    for signal in signals:
        signal_index = signal['index']
        if signal_index + num_days < len(price_data):
            entry_price = price_data[signal_index]['Close']  # Access 'Close' instead of 'close'
            exit_price = price_data[signal_index + num_days]['Close']  # Access 'Close' instead of 'close'
            profit = (exit_price - entry_price) / entry_price
            results.append({'pattern': signal['pattern'], 'profit': profit})
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_data', method=['POST'])
def returndata(ticker, start_date, end_date):
    get_data = get_stock_data(ticker, start_date, end_date)
    return get_data


@app.route('/scaled_ready', methods=['POST'])
def getAllData():
    data = request.json
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    get_data = get_stock_data(ticker, start_date, end_date)
    
    return get_data

@app.route('/stock', methods=['POST'])
def stock():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    return render_template('stock_data.html', stock_data=stock_data)



@app.route('/get_warmup_cost', methods=['GET'])
def getWarmupCost():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    stock_data = get_stock_data_visualization(ticker, start_date, end_date)
    print(stock_data)
    
    # Create traces for Open and Close prices
    trace_open = go.Scatter(x=stock_data['Date'], y=stock_data['Open'], mode='lines+markers', name='Open')
    trace_close = go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines+markers', name='Close')
    print(trace_open)
    # Create layout
    layout = go.Layout(title='Stock Prices',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'))
    
    # Create figure
    fig = go.Figure(data=[trace_open, trace_close], layout=layout)
    
    # Convert the Plotly figure to JSON
    plot_json = fig.to_json()
    plot_html = pio.to_html(fig, full_html=False)

    # Pass the JSON data to the HTML template
    return render_template('visualization.html', plot=plot_html)



@app.route('/get_sig_vars9599',methods=['GET'])
def get_sig_vars9599():
    data = request.json   #
    signals = data.get('signals', [])
    parallel_computations = data.get('parallel_computations', [])

    var_values = {}
    for signal in signals:
        values = []
        for computation in parallel_computations:
            values.extend(computation.get(signal, []))
        var_95 = calculate_var(values, 0.95)
        var_99 = calculate_var(values, 0.99)

        var_values[signal] = {'95th_percentile_var': var_95, '99th_percentile_var': var_99}

    return jsonify(var_values)


@app.route('/get_avg_vars9599', methods=['GET'])
def get_avg_vars9599():
    data = request.json  
    signals = data.get('signals', [])
    parallel_computations = data.get('parallel_computations', [])

    all_values = []
    for signal in signals:
        for computation in parallel_computations:
          
            all_values.extend(computation.get(signal, []))
# Calculate VaR values for 95% and 99% confidence levels
    var_95 = calculate_var(all_values, 0.95)
    var_99 = calculate_var(all_values, 0.99)

    return jsonify({'var95': var_95, 'var99': var_99})



@app.route('/get_sig_profit_loss',methods=['GET'])
def get_sig_profit_loss():
    data = request.json  # Assuming the data is provided in JSON format
    signals = data.get('signals', [])
    parallel_computations = data.get('parallel_computations', [])

    all_profit_loss_values = []
    for signal in signals:
        signal_profit_loss_values = []
        for computation in parallel_computations:
            # Assuming each computation provides profit/loss values for the signal
            signal_profit_loss_values.extend(computation.get(signal, []))
        all_profit_loss_values.append(signal_profit_loss_values)

    return jsonify({'profit_loss': all_profit_loss_values})
 

@app.route('/get_tot_profit_loss',methods=['GET'])
def get_tot_profit_loss():
    data = request.json  # Assuming the data is provided in JSON format
    parallel_computations = data.get('parallel_computations', [])

    total_profit_loss = 0
    for computation in parallel_computations:
        # Assuming each computation provides the total resulting profit/loss
        total_profit_loss += computation.get('resulting_profit_loss', 0)

    return jsonify({'total_profit_loss': total_profit_loss})

@app.route('/cleanup', methods=['GET'])
def cleanup():
    # Reset any data or state here as necessary
    # Example: Resetting the warmed-up scale
    global warmed_up_scale
    warmed_up_scale = None

    return jsonify({'result': 'ok'})

@app.route('/terminate', methods=['GET'])
def terminate():
    # Terminate the application, scaling it to zero
    os._exit(0)

    return jsonify({'result': 'ok'})


@app.route('/scaled_terminated', methods=['GET'])
def scaled_terminated():
    global terminated_flag
    if terminated_flag:
        return jsonify({'terminated': True})
    else:
        return jsonify({'terminated': False})

@app.route('/get_chart_url', methods=['GET'])
def get_chart_url():
    data = request.json  
    var_95 = data.get('var_95')
    var_99 = data.get('var_99')

    # Generate the chart
    chart_filename = generate_chart(var_95, var_99)
    chart_abs_path = os.path.abspath(chart_filename)
    chart_url = f'file://{chart_abs_path}'

    return jsonify({'chart_url': chart_url})

@app.route('/get_time_cost',methods=['GET'])
def get_time_cost():
    data = request.json   
    total_billable_time = data.get('total_billable_time')
    cost_per_hour = data.get('cost_per_hour')
    total_cost = total_billable_time * cost_per_hour

    return jsonify({'total_billable_time': total_billable_time, 'total_cost': total_cost})



@app.route('/get_audit', methods=['GET'])
def get_audit():
    return jsonify(previous_audits)

@app.route('/visualize', methods=['POST'])
def visualize():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    stock_data = get_stock_data_visualization(ticker, start_date, end_date)
    print(stock_data)
    
    # Create traces for Open and Close prices
    trace_open = go.Scatter(x=stock_data['Date'], y=stock_data['Open'], mode='lines+markers', name='Open')
    trace_close = go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines+markers', name='Close')
    print(trace_open)
    # Create layout
    layout = go.Layout(title='Stock Prices',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'))
    
    # Create figure
    fig = go.Figure(data=[trace_open, trace_close], layout=layout)
    
    # Convert the Plotly figure to JSON
    plot_json = fig.to_json()
    plot_html = pio.to_html(fig, full_html=False)

    # Pass the JSON data to the HTML template
    return render_template('visualization.html', plot=plot_html)

@app.route('/other')
def other():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    ohlc_data = get_stock_data(ticker, start_date, end_date)
    signals = identify_trading_signals(ohlc_data)
    monte_carlo_results = monte_carlo_analysis(ohlc_data, num_simulations=1000, confidence_levels=[0.95, 0.99])
    profitability_results = assess_profitability(ohlc_data, signals, num_days=7)
    return render_template('other.html', signals=signals, monte_carlo_results=monte_carlo_results,
                           profitability_results=profitability_results)

@app.route('/signals')
def signals():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    ohlc_data = get_stock_data(ticker, start_date, end_date)
    signals = identify_trading_signals(ohlc_data)

    # Create a plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the OHLC data
    fig.add_trace(go.Candlestick(x=[data['Date'] for data in ohlc_data],
                                  open=[data['Open'] for data in ohlc_data],
                                  high=[data['High'] for data in ohlc_data],
                                  low=[data['Low'] for data in ohlc_data],
                                  close=[data['Close'] for data in ohlc_data],
                                  name='OHLC'), secondary_y=False)

    # Plot the trading signals
    for signal in signals:
        fig.add_trace(go.Scatter(x=[ohlc_data[signal['index']]['Date']],
                                  y=[ohlc_data[signal['index']]['Close']],
                                  mode='markers',
                                  marker=dict(symbol='triangle-up' if signal['pattern'] == 'Three White Soldiers' else 'triangle-down',
                                              size=10,
                                              color='green' if signal['pattern'] == 'Three White Soldiers' else 'red'),
                                  name=signal['pattern']), secondary_y=False)

    # Update layout
    fig.update_layout(title='Trading Signals',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend_title='Patterns')

    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False)

    # Render the template with the HTML plot
    return render_template('signals.html', plot=plot_html)



@app.route('/get_warmup_cost', methods=['GET'])
def get_warmup_cost():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    total_days, total_cost = calculate_warmup_cost(start_date, end_date)

    return json.dumps({'total_days': total_days, 'total_cost': total_cost})

@app.route('/monte_carlo')
def monte_carlo():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    price_data = get_stock_data(ticker, start_date, end_date)
    num_simulations = 1000
    confidence_levels = [0.95, 0.99]

    results = monte_carlo_analysis(price_data, num_simulations, confidence_levels)

    # Convert confidence levels from strings to floats
    results = {float(key.replace('%', '')) / 100: value for key, value in results.items()}

    # Create traces for each confidence level
    traces = []
    for level, data in results.items():
        trace = go.Scatter(x=np.arange(len(data)), y=data, mode='lines', name=f'{int(level*100)}% Confidence')
        traces.append(trace)

    # Create layout
    layout = go.Layout(title='Monte Carlo Analysis',
                       xaxis=dict(title='Simulation'),
                       yaxis=dict(title='Price Change'))

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False)

    # Render the template with the HTML plot
    return render_template('monte_carlo.html', plot=plot_html)


@app.route('/profitability')
def profitability():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    ohlc_data = get_stock_data(ticker, start_date, end_date)
    signals = identify_trading_signals(ohlc_data)
    num_days = 7
    profitability_results = assess_profitability(ohlc_data, signals, num_days)

    # Extract patterns and profits for plotting
    patterns = [result['pattern'] for result in profitability_results]
    profits = [result['profit'] for result in profitability_results]

    # Create a bar chart
    fig = go.Figure(data=[go.Bar(x=patterns, y=profits)])
    fig.update_layout(title='Profitability of Trading Patterns',
                      xaxis_title='Trading Pattern',
                      yaxis_title='Profit',
                      template='plotly_dark')  # Set a dark theme for better visualization

    # Convert the figure to HTML
    plot_html = pio.to_html(fig, full_html=False)

    # Render the template with the HTML plot
    return render_template('profitability.html', plot=plot_html)



if __name__ == '__main__':
    app.run(debug=True)