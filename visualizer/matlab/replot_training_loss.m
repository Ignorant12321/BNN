% 用 MATLAB 重新绘制训练损失变化曲线，并导出可编辑的 .fig 文件。
% 使用方法：优先修改下面“配置区”的内容，然后在 MATLAB 中运行本文件。

%% 配置区
% 1. 模型输出根目录：一般只需要改模型名，不需要改 outputs/train。
modelRunRoot = fullfile("outputs", "train", "pv_usibnn_recursive");

% 2. 运行时间戳：留空 "" 时，自动使用 modelRunRoot 下最新的时间戳目录。
%    例如要固定某次运行，可以写成 "20260607-225300"。
runTimestamp = "";

% 3. 训练历史 CSV：单模型训练通常是 epoch_history.csv。
historyCsv = "epoch_history.csv";

% 4. 图表文字。
figureTitle = "模型训练损失变化曲线";
xLabelText = "迭代次数";
yLabelText = "损失值";

% 5. best epoch 的来源：留空 "" 时优先使用 monitorMetric 自动计算。
%    BNN 递归训练默认早停监控 val_generation_nrmse，但不额外绘制该曲线。
monitorMetric = "val_generation_nrmse";
bestEpochOverride = [];

% 6. 曲线和网格显示开关。
showValidationLoss = true;
showGrid = true;

% 7. 保存开关：fig 可在 MATLAB 中继续编辑，png/pdf 适合放入论文或报告。
saveMatlabFig = true;
savePng = false;
savePdf = false;

% 8. 输出文件名：默认生成 loss_curve_matlab.fig。
outputName = "loss_curve_matlab";

% 9. 样式设置：颜色使用 RGB，数值范围 0 到 1。
trainColor = [0.15 0.39 0.92];
valColor = [0.86 0.15 0.15];
bestEpochColor = [0.09 0.64 0.29];
earlyStopColor = [0.86 0.15 0.15];
lineWidth = 1.8;
markerLineWidth = 1.3;
englishFontName = "Times New Roman";
chineseFontName = "SimSun";
fontSize = 18;

%% 自动定位项目根目录、最新 run 和 CSV
% 脚本可以从任意 MATLAB 当前目录运行，因为这里会基于本文件位置定位仓库根目录。
scriptDir = fileparts(mfilename("fullpath"));
repoRoot = fullfile(scriptDir, "..", "..");
modelRunRootPath = fullfile(repoRoot, modelRunRoot);

if strlength(runTimestamp) == 0
    runTimestamp = findLatestRunTimestamp(modelRunRootPath);
    fprintf("未填写时间戳，自动使用最新运行：%s\n", runTimestamp);
end

runDir = fullfile(modelRunRootPath, runTimestamp);
csvPath = fullfile(runDir, historyCsv);
figuresDir = fullfile(runDir, "figures");

if ~isfile(csvPath)
    error("找不到训练历史 CSV：%s", csvPath);
end

if ~isfolder(figuresDir)
    mkdir(figuresDir);
end

%% 读取 CSV 并绘图
T = readtable(csvPath, "TextType", "string");
requiredColumns = ["epoch", "loss"];
missingColumns = setdiff(requiredColumns, string(T.Properties.VariableNames));
if ~isempty(missingColumns)
    error("训练历史 CSV 缺少列：%s", strjoin(missingColumns, ", "));
end

epoch = numericColumn(T, "epoch");
trainLoss = numericColumn(T, "loss");
valLoss = [];
if showValidationLoss && ismember("val_loss", string(T.Properties.VariableNames))
    valLoss = numericColumn(T, "val_loss");
end

bestEpoch = resolveBestEpoch(T, epoch, monitorMetric, bestEpochOverride);
earlyStopEpoch = resolveEarlyStopEpoch(T, epoch);

plotTrainingLoss( ...
    epoch, trainLoss, valLoss, bestEpoch, earlyStopEpoch, ...
    figureTitle, xLabelText, yLabelText, showGrid, ...
    trainColor, valColor, bestEpochColor, earlyStopColor, ...
    lineWidth, markerLineWidth, englishFontName, chineseFontName, fontSize);

saveTrainingLossFigure(gcf, figuresDir, outputName, saveMatlabFig, savePng, savePdf);

%% 本地函数
function latestTimestamp = findLatestRunTimestamp(modelRunRootPath)
    % 查找模型输出目录下最新的时间戳子目录。
    if ~isfolder(modelRunRootPath)
        error("找不到模型输出目录：%s", modelRunRootPath);
    end

    items = dir(modelRunRootPath);
    isValidDir = [items.isdir] & ~ismember({items.name}, {'.', '..'});
    runDirs = items(isValidDir);

    if isempty(runDirs)
        error("模型输出目录下没有运行时间戳目录：%s", modelRunRootPath);
    end

    names = sort(string({runDirs.name}));
    latestTimestamp = names(end);
end

function values = numericColumn(T, columnName)
    % CSV 里数值可能被读成文本，这里统一转 double。
    rawValues = T.(columnName);
    if isnumeric(rawValues)
        values = double(rawValues);
    else
        values = str2double(string(rawValues));
    end

    if any(isnan(values))
        error("列 %s 中存在无法转换为数值的单元格。", columnName);
    end
end

function bestEpoch = resolveBestEpoch(T, epoch, monitorMetric, bestEpochOverride)
    % best epoch 只用于标记，不额外绘制监控指标曲线。
    if ~isempty(bestEpochOverride)
        bestEpoch = double(bestEpochOverride);
        return;
    end

    columnNames = string(T.Properties.VariableNames);
    if strlength(monitorMetric) > 0 && ismember(monitorMetric, columnNames)
        monitorValues = numericColumn(T, monitorMetric);
    elseif ismember("val_loss", columnNames)
        monitorValues = numericColumn(T, "val_loss");
    else
        monitorValues = numericColumn(T, "loss");
    end

    [~, bestIndex] = min(monitorValues);
    bestEpoch = epoch(bestIndex);
end

function earlyStopEpoch = resolveEarlyStopEpoch(T, epoch)
    % early stop 标记来自 epoch_history.csv 的 early_stop 列。
    earlyStopEpoch = [];
    if ~ismember("early_stop", string(T.Properties.VariableNames))
        return;
    end

    rawValues = T.early_stop;
    if isnumeric(rawValues)
        values = double(rawValues);
    else
        values = str2double(string(rawValues));
    end

    markerIndex = find(~isnan(values) & values ~= 0, 1, "first");
    if ~isempty(markerIndex)
        earlyStopEpoch = epoch(markerIndex);
    end
end

function plotTrainingLoss( ...
    epoch, trainLoss, valLoss, bestEpoch, earlyStopEpoch, ...
    figureTitle, xLabelText, yLabelText, showGrid, ...
    trainColor, valColor, bestEpochColor, earlyStopColor, ...
    lineWidth, markerLineWidth, englishFontName, chineseFontName, fontSize)
    % 绘制训练损失和验证损失，保留 best epoch / early stop 标记。
    fig = figure("Name", char(figureTitle));
    hold on;

    plot(epoch, trainLoss, "-", ...
        "Color", trainColor, ...
        "LineWidth", lineWidth, ...
        "DisplayName", "train loss");

    if ~isempty(valLoss)
        plot(epoch, valLoss, "-", ...
            "Color", valColor, ...
            "LineWidth", lineWidth, ...
            "DisplayName", "val loss");
    end

    if ~isempty(bestEpoch)
        xline(bestEpoch, "--", ...
            "Color", bestEpochColor, ...
            "LineWidth", markerLineWidth, ...
            "DisplayName", "best epoch");
    end

    if ~isempty(earlyStopEpoch)
        xline(earlyStopEpoch, ":", ...
            "Color", earlyStopColor, ...
            "LineWidth", markerLineWidth, ...
            "DisplayName", "early stop");
    end

    if showGrid
        grid on;
    else
        grid off;
    end

    title(figureTitle, "FontName", chooseTextFont(figureTitle, englishFontName, chineseFontName));
    xlabel(xLabelText, "FontName", chooseTextFont(xLabelText, englishFontName, chineseFontName));
    ylabel(yLabelText, "FontName", chooseTextFont(yLabelText, englishFontName, chineseFontName));
    lgd = legend("Location", "best");
    lgd.FontName = englishFontName;
    lgd.FontSize = fontSize;

    ax = gca;
    ax.FontName = englishFontName;
    ax.FontSize = fontSize;
    ax.LineWidth = 1;
    box on;
    hold off;

    fig.Position(3:4) = [920, 420];
end

function saveTrainingLossFigure(figHandle, figuresDir, outputName, saveMatlabFig, savePng, savePdf)
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
