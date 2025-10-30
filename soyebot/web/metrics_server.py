"""Lightweight Flask metrics dashboard for SoyeBot.

Designed for 1GB RAM environments - minimal dependencies, efficient rendering.
"""

import logging
from flask import Flask, render_template, jsonify
from threading import Thread
from metrics import get_metrics

logger = logging.getLogger(__name__)

# Create Flask app with minimal configuration
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Reduce memory


@app.route('/')
def index():
    """Render main metrics dashboard."""
    return render_template('metrics.html')


@app.route('/api/metrics')
def api_metrics():
    """JSON endpoint for metrics data.

    Returns all metrics in JSON format for programmatic access or AJAX updates.
    """
    try:
        metrics = get_metrics()
        summary = metrics.get_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def api_health():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok'})


def run_metrics_server(host: str = '0.0.0.0', port: int = 5000):
    """Run Flask metrics server.

    Args:
        host: Host to bind to (default: 0.0.0.0 for external access)
        port: Port to bind to (default: 5000)
    """
    logger.info(f"Starting metrics server on {host}:{port}")

    # Disable Flask's default logger spam
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Metrics server error: {e}", exc_info=True)


def start_metrics_server_background(host: str = '0.0.0.0', port: int = 5000):
    """Start metrics server in background thread.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        Thread object for the metrics server
    """
    thread = Thread(
        target=run_metrics_server,
        args=(host, port),
        daemon=True,  # Daemon thread exits when main program exits
        name='MetricsServer'
    )
    thread.start()
    logger.info(f"Metrics server started in background thread (http://{host}:{port})")
    return thread
