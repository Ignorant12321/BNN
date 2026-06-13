% 用 MATLAB 绘制 4h BNN 多步预测策略的逐步长误差折线图，并导出可编辑的 .fig 文件。
% 使用方法：先运行 python -m src.experiments.compare_bnn_strategies_4h，再在 MATLAB 中运行本文件。

%% 配置区
% 1. 对比输出根目录：多步策略命令会写到 outputs/comparisons。
comparisonRoot = fullfile("outputs", "comparisons");

% 2. 对比目录名：留空 "" 时，自动使用最新的 bnn_4h_strategies_* 目录。
%    例如要固定某次运行，可以写成 "bnn_4h_strategies_20260527-122846"。
comparisonName = "";
comparisonPrefix = "bnn_4h_strategies_";

% 3. 预测 CSV 目录：多步策略对比会写出 predictions/Direct.csv 等文件。
predictionsFolder = "predictions";

% 4. 要绘制的逐步长指标。第一列是指标名，第二列是纵轴标题。
%    指标名可选 "mae"、"rmse"、"nmae"、"nrmse"。
metricsToPlot = [
    "mae", "MAE"
    "rmse", "RMSE"
    "nmae", "NMAE"
    "nrmse", "NRMSE"
];

% 5. NRMSE/NMAE 显示为百分比；如果想显示小数，把这里改成 false。
showPercentForNormalizedMetrics = true;

% 6. 是否只统计发电时段。false 表示使用全部测试样本；true 表示只保留 06:00-18:00。
useGenerationPeriodOnly = false;
generationStart = "06:00";
generationEnd = "18:00";

% 7. 策略显示顺序和图例文字。
strategyOrder = ["Direct", "Recursive", "MIMO"];
legendLabels = [
    "Direct（直接策略）"
    "Recursive（递推策略）"
    "MIMO（多输出策略）"
];

% 8. 图表文字。
figureTitle = "";
xLabelText = "预测步长（15分钟/步）";

% 9. 保存开关：fig 可在 MATLAB 中继续编辑，png/pdf 适合放入论文或报告。
saveMatlabFig = true;
savePng = false;
savePdf = false;

% 10. 输出文件名前缀：默认生成 strategy_horizon_mae_matlab.fig、strategy_horizon_nrmse_matlab.fig 等文件。
outputPrefix = "strategy_horizon_";
outputSuffix = "_matlab";

% 11. 样式设置。
lineColors = [
    0.72 0.10 0.10
    0.13 0.40 0.86
    0.08 0.50 0.25
];
lineStyles = ["--", "-.", ":"];
markers = ["o", "s", "^"];
lineWidth = 1.8;
markerSize = 5.5;
showGrid = true;
englishFontName = "Times New Roman";
chineseFontName = "SimSun";
fontSize = 18;

%% 自动定位项目根目录、最新 comparison 和预测目录
scriptDir = fileparts(mfilename("fullpath"));
repoRoot = fullfile(scriptDir, "..", "..");
comparisonRootPath = fullfile(repoRoot, comparisonRoot);

if strlength(comparisonName) == 0
    comparisonName = findLatestComparisonName(comparisonRootPath, comparisonPrefix);
    fprintf("未填写对比目录，自动使用最新多步策略对比：%s\n", comparisonName);
end

comparisonDir = fullfile(comparisonRootPath, comparisonName);
predictionsDir = fullfile(comparisonDir, predictionsFolder);
figuresDir = fullfile(comparisonDir, "figures");

if ~isfolder(predictionsDir)
    error("找不到多步策略预测目录：%s", predictionsDir);
end

if ~isfolder(figuresDir)
    mkdir(figuresDir);
end

%% 读取预测 CSV
predictionTables = cell(numel(strategyOrder), 1);
for i = 1:numel(strategyOrder)
    strategy = strategyOrder(i);
    csvPath = fullfile(predictionsDir, strategy + ".csv");
    if ~isfile(csvPath)
        error("找不到策略预测 CSV：%s", csvPath);
    end

    T = readtable(csvPath, "TextType", "string");
    if useGenerationPeriodOnly
        T = filterGenerationPeriod(T, generationStart, generationEnd);
    end
    predictionTables{i} = T;
end

%% 批量计算逐步长指标并绘图
for metricIndex = 1:size(metricsToPlot, 1)
    metricName = metricsToPlot(metricIndex, 1);
    yLabelText = metricsToPlot(metricIndex, 2);
    formatYAxisAsPercent = isNormalizedMetric(metricName) && showPercentForNormalizedMetrics;

    series = struct("label", {}, "horizon", {}, "value", {});
    for i = 1:numel(strategyOrder)
        [horizons, values] = horizonMetric(predictionTables{i}, metricName, showPercentForNormalizedMetrics);
        series(i).label = legendLabels(i);
        series(i).horizon = horizons;
        series(i).value = values;

        fprintf("%s：已计算 %s，共 %d 个预测步长。\n", strategyOrder(i), metricName, numel(horizons));
    end

    plotStrategyHorizonLines(series, figureTitle, xLabelText, yLabelText, ...
        lineColors, lineStyles, markers, lineWidth, markerSize, showGrid, formatYAxisAsPercent, ...
        englishFontName, chineseFontName, fontSize);
    saveHorizonFigure(gcf, figuresDir, outputPrefix + metricName + outputSuffix, saveMatlabFig, savePng, savePdf);
end

%% 本地函数
function latestName = findLatestComparisonName(comparisonRootPath, comparisonPrefix)
    % 查找最新的多步策略对比目录。
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

function T = filterGenerationPeriod(T, generationStart, generationEnd)
    % 只保留指定钟点范围内的 target_time。
    if ~ismember("target_time", string(T.Properties.VariableNames))
        error("预测 CSV 缺少 target_time 列，无法筛选发电时段。");
    end

    targetTime = datetime(T.target_time, "InputFormat", "yyyy-MM-dd HH:mm:ss");
    clockMinutes = hour(targetTime) * 60 + minute(targetTime);
    startMinutes = clockToMinutes(generationStart);
    endMinutes = clockToMinutes(generationEnd);
    T = T(clockMinutes >= startMinutes & clockMinutes <= endMinutes, :);
end

function [horizons, values] = horizonMetric(T, metricName, showPercentForNormalizedMetrics)
    % 逐 horizon 计算 MAE/RMSE/NMAE/NRMSE。
    requiredColumns = ["horizon", "target", "mean"];
    missingColumns = setdiff(requiredColumns, string(T.Properties.VariableNames));
    if ~isempty(missingColumns)
        error("预测 CSV 缺少列：%s", strjoin(missingColumns, ", "));
    end

    horizonValues = unique(T.horizon);
    horizonValues = sort(horizonValues(:));
    horizons = horizonValues + 1;
    values = zeros(size(horizonValues));

    for index = 1:numel(horizonValues)
        H = T(T.horizon == horizonValues(index), :);
        target = double(H.target);
        predicted = double(H.mean);
        errorValues = predicted - target;
        mae = mean(abs(errorValues), "omitnan");
        rmse = sqrt(mean(errorValues .^ 2, "omitnan"));
        scale = normalizationScale(target);

        switch lower(metricName)
            case "mae"
                value = mae;
            case "rmse"
                value = rmse;
            case "nmae"
                value = mae / scale;
            case "nrmse"
                value = rmse / scale;
            otherwise
                error("不支持的指标：%s。可选 mae/rmse/nmae/nrmse。", metricName);
        end

        if isNormalizedMetric(metricName) && showPercentForNormalizedMetrics
            value = value * 100;
        end
        values(index) = value;
    end
end

function scale = normalizationScale(target)
    % 与 Python 的 normalization_scale 逻辑保持一致。
    target = target(~isnan(target));
    if isempty(target)
        scale = 1.0;
        return;
    end

    targetRange = max(target) - min(target);
    if targetRange > 0
        scale = targetRange;
        return;
    end

    meanAbs = mean(abs(target));
    if meanAbs > 0
        scale = meanAbs;
        return;
    end

    scale = 1.0;
end

function result = isNormalizedMetric(metricName)
    result = any(strcmpi(metricName, ["nmae", "nrmse"]));
end

function minutesValue = clockToMinutes(value)
    parts = split(string(value), ":");
    minutesValue = str2double(parts(1)) * 60 + str2double(parts(2));
end

function plotStrategyHorizonLines(series, figureTitle, xLabelText, yLabelText, ...
    lineColors, lineStyles, markers, lineWidth, markerSize, showGrid, formatYAxisAsPercent, ...
    englishFontName, chineseFontName, fontSize)
    % 绘制 Direct/Recursive/MIMO 逐步长折线。
    fig = figure("Name", "Strategy horizon " + yLabelText);
    hold on;

    for index = 1:numel(series)
        color = lineColors(mod(index - 1, size(lineColors, 1)) + 1, :);
        plot(series(index).horizon, series(index).value, ...
            "LineStyle", lineStyles(index), ...
            "Marker", markers(index), ...
            "Color", color, ...
            "MarkerFaceColor", color, ...
            "MarkerEdgeColor", color, ...
            "LineWidth", lineWidth, ...
            "MarkerSize", markerSize);
    end

    if strlength(figureTitle) > 0
        title(figureTitle, "FontName", chooseTextFont(figureTitle, englishFontName, chineseFontName));
    end

    xlabel(xLabelText, "FontName", chooseTextFont(xLabelText, englishFontName, chineseFontName));
    ylabel(yLabelText, "FontName", chooseTextFont(yLabelText, englishFontName, chineseFontName));
    lgd = legend(string({series.label}), "Location", "best");
    lgd.FontName = englishFontName;
    lgd.FontSize = fontSize;

    if showGrid
        grid on;
    else
        grid off;
    end

    ax = gca;
    ax.FontName = englishFontName;
    ax.FontSize = fontSize;
    ax.LineWidth = 1;
    allHorizons = vertcat(series.horizon);
    maxHorizon = max(allHorizons);
    ax.XLim = [1, maxHorizon];
    ax.XTick = 1:maxHorizon;
    if formatYAxisAsPercent
        ax.YTickLabel = compose("%.1f%%", ax.YTick);
    end
    box on;
    hold off;

    fig.Position(3:4) = [900, 460];
end

function saveHorizonFigure(figHandle, figuresDir, outputName, saveMatlabFig, savePng, savePdf)
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
