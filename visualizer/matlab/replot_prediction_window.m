% 用 MATLAB 重新绘制 Python 生成的预测 CSV，方便后续手动编辑坐标轴、标题、颜色和图例。
% 使用方法：优先修改下面“配置区”的内容，然后在 MATLAB 中运行本文件。

%% 配置区
% 1. 模型输出根目录：一般只需要改模型名，不需要改 outputs/train。
modelRunRoot = fullfile("outputs", "train", "pv_usibnn_recursive");

% 2. 运行时间戳：留空 "" 时，自动使用 modelRunRoot 下最新的时间戳目录。
%    例如要固定某次运行，可以写成 "20260607-171459"。
runTimestamp = "";

% 3. CSV 文件名：单模型训练通常是 predictions/test.csv。
predictionCsv = fullfile("predictions", "test.csv");

% 4. 一次绘制多个预测轨迹：每一行是 [起报时, 起报分, 结束时, 结束分]。
timeWindows = [
    6,  0, 10,  0
    10, 0, 14,  0
    14, 0, 18,  0
];

% 5. 起报日期：留空 "" 时，自动选择该起报时刻内第一个有数据的日期。
%    例如要固定画 2020 年 6 月 14 日，可以写成 "2020-06-14"。
plotDate = "";

% 6. 图表文字：窗口时间会自动拼到标题里。
titlePrefix = "Prediction";
xLabelText = "Time";
yLabelText = "AC Power / kW";

% 7. 曲线和区间显示开关。
showInterval95 = true;
showInterval90 = true;
showGrid = true;

% 8. 保存开关：fig 可在 MATLAB 中继续编辑，png/pdf 适合放入论文或报告。
saveMatlabFig = true;
savePng = false;
savePdf = false;

% 9. 输出文件名前缀：最终会生成 prediction_0600_1000_matlab.fig 这种文件名。
outputSuffix = "_matlab";

% 10. 样式设置：颜色使用 RGB，数值范围 0 到 1。
actualColor = [0.00 0.00 0.00];
predictionColor = [1.00 0.00 0.00];
interval95Color = [0.55 0.95 0.95];
interval90Color = [0.45 0.70 0.95];
actualLineStyle = "-";
predictionLineStyle = "--";
lineWidth = 1.8;
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
csvPath = fullfile(runDir, predictionCsv);

if ~isfile(csvPath)
    error("找不到预测 CSV：%s", csvPath);
end

%% 读取 CSV
T = readtable(csvPath);
T.target_time = datetime(T.target_time, "InputFormat", "yyyy-MM-dd HH:mm:ss");

%% 批量绘制三个时间窗口
for i = 1:size(timeWindows, 1)
    startClock = duration(timeWindows(i, 1), timeWindows(i, 2), 0);
    endClock = duration(timeWindows(i, 3), timeWindows(i, 4), 0);
    windowLabel = formatWindowLabel(startClock, endClock);

    plotPredictionWindow( ...
        T, startClock, endClock, plotDate, ...
        titlePrefix + " " + windowLabel, xLabelText, yLabelText, ...
        showInterval95, showInterval90, showGrid, ...
        actualColor, predictionColor, interval95Color, interval90Color, ...
        actualLineStyle, predictionLineStyle, lineWidth, englishFontName, chineseFontName, fontSize);

    savePredictionFigure( ...
        gcf, runDir, "prediction_" + formatWindowSlug(startClock, endClock) + outputSuffix, ...
        saveMatlabFig, savePng, savePdf);
end

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

function plotPredictionWindow( ...
    T, startClock, endClock, plotDate, figureTitle, xLabelText, yLabelText, ...
    showInterval95, showInterval90, showGrid, ...
    actualColor, predictionColor, interval95Color, interval90Color, ...
    actualLineStyle, predictionLineStyle, lineWidth, englishFontName, chineseFontName, fontSize)
    % 根据 horizon 反推起报时刻，只保留同一次未来 4h 预测轨迹。
    if ~ismember("horizon", string(T.Properties.VariableNames))
        error("预测 CSV 缺少 horizon 列，无法还原同一起报时刻的未来预测轨迹。");
    end

    issueTime = T.target_time - minutes((T.horizon + 1) * 15);
    issueDates = unique(dateshift(issueTime, "start", "day"));

    if strlength(plotDate) == 0
        % 与 Python 绘图逻辑保持一致：不指定日期时，选择第一个包含该起报时刻的日期。
        selectedDate = NaT;
        for dateIndex = 1:numel(issueDates)
            candidateIssueTime = issueDates(dateIndex) + startClock;
            candidateRows = issueTime == candidateIssueTime;
            if any(candidateRows)
                selectedDate = issueDates(dateIndex);
                break;
            end
        end
        if isnat(selectedDate)
            error("没有找到起报时刻为 %s 的预测数据。", string(startClock));
        end
        dateModeText = "自动";
    else
        % 指定日期时，只画这一天的该起报时刻；如果没有数据，会给出明确错误。
        selectedDate = datetime(plotDate, "InputFormat", "yyyy-MM-dd");
        dateModeText = "指定";
    end

    selectedIssueTime = selectedDate + startClock;
    selectedEndTime = selectedDate + endClock;
    W = T(issueTime == selectedIssueTime & T.target_time > selectedIssueTime & T.target_time <= selectedEndTime, :);

    if isempty(W)
        error("日期 %s 起报时刻 %s 到 %s 之间没有预测数据。", ...
            datestr(selectedDate, "yyyy-mm-dd"), string(startClock), string(endClock));
    end

    W = sortrows(W, {'target_time', 'label'});

    % 同一个 target_time 可能来自多个样本或 horizon，这里按时间求均值后再画。
    G = groupsummary(W, "target_time", "mean", ...
        ["target", "mean", "lower_90", "upper_90", "lower_95", "upper_95"]);

    x = G.target_time;
    actual = G.mean_target;
    predicted = G.mean_mean;
    lower90 = G.mean_lower_90;
    upper90 = G.mean_upper_90;
    lower95 = G.mean_lower_95;
    upper95 = G.mean_upper_95;

    fprintf("%s：%s选择起报 %s %s，共 %d 个时间点，范围 %s 到 %s。\n", ...
        char(figureTitle), char(dateModeText), datestr(selectedDate, "yyyy-mm-dd"), ...
        char(string(startClock)), height(G), datestr(min(x), "HH:MM"), datestr(max(x), "HH:MM"));

    figure("Name", char(figureTitle));
    hold on;

    legendItems = strings(0);

    if showInterval95
        fill([x; flipud(x)], [lower95; flipud(upper95)], ...
            interval95Color, "FaceAlpha", 0.35, "EdgeColor", "none");
        legendItems(end + 1) = "95% interval";
    end

    if showInterval90
        fill([x; flipud(x)], [lower90; flipud(upper90)], ...
            interval90Color, "FaceAlpha", 0.25, "EdgeColor", "none");
        legendItems(end + 1) = "90% interval";
    end

    plot(x, actual, "LineStyle", actualLineStyle, "Color", actualColor, "LineWidth", lineWidth);
    legendItems(end + 1) = "Actual";

    plot(x, predicted, "LineStyle", predictionLineStyle, "Color", predictionColor, "LineWidth", lineWidth);
    legendItems(end + 1) = "Prediction";

    if showGrid
        grid on;
    else
        grid off;
    end

    xlabel(xLabelText, "FontName", chooseTextFont(xLabelText, englishFontName, chineseFontName));
    ylabel(yLabelText, "FontName", chooseTextFont(yLabelText, englishFontName, chineseFontName));
    title(figureTitle, "FontName", chooseTextFont(figureTitle, englishFontName, chineseFontName));
    lgd = legend(legendItems, "Location", "best");
    lgd.FontName = englishFontName;
    lgd.FontSize = fontSize;

    ax = gca;
    xlim([selectedIssueTime + minutes(15), selectedEndTime]);
    ax.FontName = englishFontName;
    ax.FontSize = fontSize;
    ax.LineWidth = 1;
    box on;
    hold off;
end

function savePredictionFigure(figHandle, runDir, outputName, saveMatlabFig, savePng, savePdf)
    % 根据配置保存 fig/png/pdf。
    outputBase = fullfile(runDir, "predictions", outputName);

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

function label = formatWindowLabel(startClock, endClock)
    % 生成图标题中的时间窗口，例如 06:00-10:00。
    label = string(compose("%02d:%02d-%02d:%02d", ...
        hours(startClock), minutes(startClock) - hours(startClock) * 60, ...
        hours(endClock), minutes(endClock) - hours(endClock) * 60));
end

function slug = formatWindowSlug(startClock, endClock)
    % 生成文件名中的时间窗口，例如 0600_1000。
    slug = string(compose("%02d%02d_%02d%02d", ...
        hours(startClock), minutes(startClock) - hours(startClock) * 60, ...
        hours(endClock), minutes(endClock) - hours(endClock) * 60));
end
