from pathlib import Path


def test_visualizer_sidebar_has_read_create_nav_tabs():
    html = Path("visualizer/index.html").read_text(encoding="utf-8")

    assert 'class="sidebar-tabs"' in html
    assert 'data-sidebar-tab="read"' in html
    assert 'data-sidebar-tab="create"' in html
    assert 'data-sidebar-tab="nav"' in html
    assert 'id="readControls"' in html
    assert 'id="createControls"' in html
    assert 'id="navControls"' in html


def test_visualizer_navigation_tab_has_section_directory():
    html = Path("visualizer/index.html").read_text(encoding="utf-8")

    assert 'id="sectionDirectory"' in html
    for section_id in ["statsSection", "metricSection", "predictionSection", "configSection", "figureSection"]:
        assert f'id="{section_id}"' in html
        assert f'data-nav-target="{section_id}"' in html


def test_visualizer_read_and_create_switch_to_navigation_tab():
    js = Path("visualizer/app.js").read_text(encoding="utf-8")

    assert "switchToNavigation();" in js
    assert js.count("switchToNavigation();") >= 2


def test_visualizer_figure_click_does_not_rerender_before_double_click():
    js = Path("visualizer/app.js").read_text(encoding="utf-8")

    click_block = js.split('$("figureGrid").addEventListener("click"', 1)[1].split('$("figureGrid").addEventListener("dblclick"', 1)[0]
    assert "selectFigureCard(card);" in click_block
    assert "renderFigures();" not in click_block


def test_visualizer_loss_figures_have_compact_styling():
    css = Path("visualizer/styles.css").read_text(encoding="utf-8")
    js = Path("visualizer/app.js").read_text(encoding="utf-8")

    assert "figure-group-loss" in js
    assert ".figure-group-loss" in css
    assert "max-height" in css


def test_visualizer_navigation_scrolls_vertically_without_horizontal_page_shift():
    js = Path("visualizer/app.js").read_text(encoding="utf-8")
    css = Path("visualizer/styles.css").read_text(encoding="utf-8")

    assert "function scrollToMainSection(target)" in js
    assert "window.scrollTo({" in js
    assert "left: 0" in js
    assert "target.scrollIntoView" not in js
    assert "overflow-x: hidden" in css
    assert ".panel {\n  min-width: 0;" in css


def test_visualizer_prediction_summary_displays_interval_metrics():
    js = Path("visualizer/app.js").read_text(encoding="utf-8")

    assert '["picp90", "PICP 90"]' in js
    assert '["pinaw90", "PINAW 90"]' in js
    assert '["picp95", "PICP 95"]' in js
    assert '["pinaw95", "PINAW 95"]' in js


def test_visualizer_create_closes_current_comparison_before_running_compare():
    js = Path("visualizer/app.js").read_text(encoding="utf-8")
    create_block = js.split("async function createComparison()", 1)[1].split("function closeCurrentComparison", 1)[0]

    assert "closeCurrentComparison();" in create_block
    assert create_block.index("closeCurrentComparison();") < create_block.index('fetchJson("/api/comparisons"')
    assert 'state.activeSidebarTab = "create";' in create_block


def test_visualizer_has_shared_close_current_comparison_helper():
    js = Path("visualizer/app.js").read_text(encoding="utf-8")

    assert "function closeCurrentComparison()" in js
    assert "state.currentComparison = null;" in js
    assert "state.runs = [];" in js
    assert "state.figures = [];" in js
