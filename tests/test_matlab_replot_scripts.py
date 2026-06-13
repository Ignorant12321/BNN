from pathlib import Path
import re


def test_point_error_metrics_matlab_script_exports_fig_files():
    script = Path("visualizer/matlab/replot_point_error_metrics.m")

    text = script.read_text(encoding="utf-8")

    assert "recursive_point_forecasts_4h_" in text
    assert "model_metrics.csv" in text
    assert "metricsToPlot" in text
    assert "savefig" in text
    assert "metrics_test_mae_matlab.fig" in text


def test_strategy_metric_bar_matlab_script_has_been_removed():
    assert not Path("visualizer/matlab/replot_bnn_strategy_metrics.m").exists()


def test_strategy_horizon_line_matlab_script_exports_fig_files():
    script = Path("visualizer/matlab/replot_bnn_strategy_horizon_lines.m")

    text = script.read_text(encoding="utf-8")

    assert "bnn_4h_strategies_" in text
    assert "predictions" in text
    assert "metricsToPlot" in text
    assert '"mae", "MAE"' in text
    assert '"rmse", "RMSE"' in text
    assert '"nmae", "NMAE"' in text
    assert '"nrmse", "NRMSE"' in text
    assert "plotStrategyHorizonLines" in text
    assert "savefig" in text
    assert "strategy_horizon_mae_matlab.fig" in text
    assert "strategy_horizon_nrmse_matlab.fig" in text


def test_training_loss_matlab_script_exports_editable_loss_curve():
    script = Path("visualizer/matlab/replot_training_loss.m")

    text = script.read_text(encoding="utf-8")

    assert "pv_usibnn_recursive" in text
    assert "epoch_history.csv" in text
    assert "模型训练损失变化曲线" in text
    assert "迭代次数" in text
    assert "损失值" in text
    assert "best epoch" in text
    assert "early stop" in text
    assert "savefig" in text
    assert "loss_curve_matlab.fig" in text


def test_prediction_window_matlab_script_matches_paper_style_lines():
    script = Path("visualizer/matlab/replot_prediction_window.m")

    text = script.read_text(encoding="utf-8")

    assert "predictionColor = [1.00 0.00 0.00];" in text
    assert 'predictionLineStyle = "--";' in text
    assert 'actualLineStyle = "-";' in text
    assert "interval95Color = [0.55 0.95 0.95];" in text
    assert "interval90Color = [0.45 0.70 0.95];" in text
    assert '"LineStyle", actualLineStyle' in text
    assert '"LineStyle", predictionLineStyle' in text
    assert 'legendItems(end + 1) = "95% interval";' in text
    assert 'legendItems(end + 1) = "90% interval";' in text
    assert 'legendItems(end + 1) = "Actual";' in text
    assert 'legendItems(end + 1) = "Prediction";' in text


def test_matlab_replot_scripts_use_paper_font_defaults():
    scripts = sorted(Path("visualizer/matlab").glob("replot_*.m"))

    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        match = re.search(r"fontSize\s*=\s*([\d.]+);", text)
        assert match, f"{script} should define fontSize in its configuration section"
        assert float(match.group(1)) == 10.5, f"{script} should use Chinese size 5"
        assert 'englishFontName = "Times New Roman";' in text
        assert 'chineseFontName = "SimSun";' in text
