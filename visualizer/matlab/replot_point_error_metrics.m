% 用 MATLAB 重新绘制点预测误差对比柱状图，并导出可编辑的 .fig 文件。
% 使用方法：先运行 Python 点预测对比，再在 MATLAB 中运行本文件。

%% 配置区
% 1. 对比输出根目录：点预测命令会写到 outputs/comparisons。
comparisonRoot = fullfile("outputs", "comparisons");

% 2. 对比目录名：留空 "" 时，自动使用最新的 recursive_point_forecasts_4h_* 目录。
%    例如要固定某次运行，可以写成 "recursive_point_forecasts_4h_20260607-171459"。
comparisonName = "";
comparisonPrefix = "recursive_point_forecasts_4h_";

% 3. 指标 CSV：点预测对比的汇总误差表。
metricsCsv = "model_metrics.csv";

% 4. 要重画的误差指标。第一列是 CSV 列名，第二列是图标题。
metricsToPlot = [
    % "test_mae", "Test MAE"
    % "test_rmse", "Test RMSE"
    "test_nmae", "Test NMAE"
    "test_nrmse", "Test NRMSE"
    % "test_generation_mae", "Generation MAE"
    % "test_generation_rmse", "Generation RMSE"
    % "test_generation_nmae", "Generation NMAE"
    % "test_generation_nrmse", "Generation NRMSE"
];

% 5. 保存开关：fig 可在 MATLAB 中继续编辑，png/pdf 适合放入论文或报告。
saveMatlabFig = true;
savePng = false;
savePdf = false;

% 6. 输出文件名前缀：最终会生成 metrics_test_mae_matlab.fig 这种文件名。
outputSuffix = "_matlab";

% 7. 样式设置：颜色使用 RGB，数值范围 0 到 1。
barColors = [
    0.15 0.39 0.92
    0.86 0.15 0.15
    0.09 0.64 0.29
    0.58 0.20 0.92
    0.92 0.35 0.04
    0.03 0.57 0.70
];
showGrid = true;
barWidth = 0.68;
englishFontName = "Times New Roman";
chineseFontName = "SimSun";
fontSize = 18;

%% 自动定位项目根目录、最新 comparison 和 CSV
scriptDir = fileparts(mfilename("fullpath"));
repoRoot = fullfile(scriptDir, "..", "..");
comparisonRootPath = fullfile(repoRoot, comparisonRoot);

if strlength(comparisonName) == 0
    comparisonName = findLatestComparisonName(comparisonRootPath, comparisonPrefix);
    fprintf("未填写对比目录，自动使用最新点预测对比：%s\n", comparisonName);
end

comparisonDir = fullfile(comparisonRootPath, comparisonName);
csvPath = fullfile(comparisonDir, metricsCsv);
figuresDir = fullfile(comparisonDir, "figures");

if ~isfile(csvPath)
    error("找不到点预测误差 CSV：%s", csvPath);
end

if ~isfolder(figuresDir)
    mkdir(figuresDir);
end

%% 读取 CSV 并批量绘图
T = readtable(csvPath, "TextType", "string");
requiredColumns = ["label", metricsToPlot(:, 1)'];
missingColumns = setdiff(requiredColumns, string(T.Properties.VariableNames));
if ~isempty(missingColumns)
    error("点预测误差 CSV 缺少列：%s", strjoin(missingColumns, ", "));
end

labels = string(T.label);

for i = 1:size(metricsToPlot, 1)
    metricName = metricsToPlot(i, 1);
    figureTitle = metricsToPlot(i, 2);
    values = numericMetricColumn(T, metricName);

    plotMetricBar(labels, values, figureTitle, metricName, ...
        barColors, barWidth, showGrid, englishFontName, chineseFontName, fontSize);

    saveMetricFigure(gcf, figuresDir, "metrics_" + metricName + outputSuffix, ...
        saveMatlabFig, savePng, savePdf);
end

%% 本地函数
function latestName = findLatestComparisonName(comparisonRootPath, comparisonPrefix)
    % 查找最新的点预测对比目录。
    if ~isfolder(comparisonRootPath)
        error("找不到对比输出目录：%s", comparisonRootPath);
    end

    items = dir(fullfile(comparisonRootPath, comparisonPrefix + "*"));
    isValidDir = [items.isdir];
    compareDirs = items(isValidDir);

    if isempty(compareDirs)
        error("对比输出目录下没有 %s* 目录：%s", comparisonPrefix, comparisonRootPath);
    end

    names = sort(string({compareDirs.name}));
    latestName = names(end);
end

function values = numericMetricColumn(T, metricName)
    % CSV 里指标可能被 pandas 写成数字，也可能被读成文本，这里统一转 double。
    rawValues = T.(metricName);
    if isnumeric(rawValues)
        values = double(rawValues);
    else
        values = str2double(string(rawValues));
    end

    if any(isnan(values))
        error("指标列 %s 中存在无法转换为数值的单元格。", metricName);
    end
end

function plotMetricBar(labels, values, figureTitle, metricName, ...
    barColors, barWidth, showGrid, englishFontName, chineseFontName, fontSize)
    % 按 CSV 顺序绘制模型误差柱状图。
    fig = figure("Name", char(figureTitle));
    categories = categorical(labels, labels, "Ordinal", true);
    chart = bar(categories, values, barWidth, "FaceColor", "flat");

    for index = 1:numel(values)
        chart.CData(index, :) = barColors(mod(index - 1, size(barColors, 1)) + 1, :);
    end

    title(figureTitle, "FontName", chooseTextFont(figureTitle, englishFontName, chineseFontName));
    xlabel("Model", "FontName", englishFontName);
    ylabel(metricName, "FontName", chooseTextFont(metricName, englishFontName, chineseFontName));

    if showGrid
        grid on;
    else
        grid off;
    end

    ax = gca;
    ax.FontName = englishFontName;
    ax.FontSize = fontSize;
    ax.LineWidth = 1;
    ax.YGrid = "on";
    ax.XGrid = "off";
    xtickangle(20);
    box on;

    for index = 1:numel(values)
        text(index, values(index), sprintf("%.4g", values(index)), ...
            "HorizontalAlignment", "center", ...
            "VerticalAlignment", "bottom", ...
            "FontName", englishFontName, ...
            "FontSize", max(fontSize - 2, 8));
    end

    fig.Position(3:4) = [760, 420];
end

function saveMetricFigure(figHandle, figuresDir, outputName, saveMatlabFig, savePng, savePdf)
    % 根据配置保存 fig/png/pdf。
    outputBase = fullfile(figuresDir, outputName);

    if saveMatlabFig
        figPath = outputBase + ".fig";
        savefig(figHandle, figPath);
        fprintf("已保存 MATLAB 可编辑图：%s\n", figPath);
    end

    if savePng
        pngPath = outputBase + ".png";
        exportgraphics(figHandle, pngPath, "Resolution", 300);
        fprintf("已保存 PNG：%s\n", pngPath);
    end

    if savePdf
        pdfPath = outputBase + ".pdf";
        exportgraphics(figHandle, pdfPath, "ContentType", "vector");
        fprintf("已保存 PDF：%s\n", pdfPath);
    end
end

function fontName = chooseTextFont(textValue, englishFontName, chineseFontName)
    % 中文使用宋体，英文使用 Times New Roman。
    if containsCjk(textValue)
        fontName = chineseFontName;
    else
        fontName = englishFontName;
    end
end

function result = containsCjk(textValue)
    textChars = char(string(textValue));
    result = any(textChars >= char(19968) & textChars <= char(40959));
end
