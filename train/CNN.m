% 清空环境
clear; clc;

% 加载外部数据
load("E:\work\svr\duc\mutil\20250108\tensor3\tensor_WPC.mat"); % 加载特征数据
load("E:\work\svr\duc\mutil\20250108\tensor3\WPC_MOS.mat"); % 加载分数数据

% 如果数据是表格类型，需要转换为数组
if istable(tensor)
    X = table2array(tensor);  % 将表格转换为数组
else
    X = tensor;  % 如果已经是数组，直接使用
end

if istable(third_column)
    Y = table2array(third_column);  % 将表格转换为数组
else
    Y = third_column;  % 如果已经是数组，直接使用
end

% 确保数据维度正确
if size(X, 1) ~= length(Y)
    error('特征数据样本数量与分数数据样本数量不匹配');
end

% 为每个样本添加编号
sampleIDs = (1:size(X, 1))';

% 将数据组合成表，方便统一打散
dataTable = table(sampleIDs, Y, num2cell(X, [2, 3, 4]), ...
    'VariableNames', {'SampleID', 'Score', 'FeatureTensor'});

% 随机打散
shuffleIdx = randperm(height(dataTable));
dataTable = dataTable(shuffleIdx, :);

% 设置比例和划分
trainRatio = 0.6; % 60% 训练集，40% 测试集
numTrain = floor(height(dataTable) * trainRatio);

% 划分训练集和测试集
trainTable = dataTable(1:numTrain, :);
testTable = dataTable(numTrain+1:end, :);

% 提取训练集数据
XTrain = cat(1, trainTable.FeatureTensor{:});
YTrain = trainTable.Score;

% 提取测试集数据
XTest = cat(1, testTable.FeatureTensor{:});
YTest = testTable.Score;

% 提取测试集编号
testSampleIDs = testTable.SampleID;

% 调整输入张量维度为 CNN 格式
XTrain = permute(XTrain, [2, 3, 4, 1]); % [5, 3, 3, numTrain]
XTest = permute(XTest, [2, 3, 4, 1]);   % [5, 3, 3, numTest]

% 定义网络结构
layers = [
    imageInputLayer(size(XTrain, 1:3), 'Normalization', 'zerocenter') % 输入层
    convolution2dLayer(3, 64, 'Padding', 'same') % 3x3 卷积层，64 个滤波器
    reluLayer % ReLU 激活层chroma
    maxPooling2dLayer(2, 'Stride', 1) % 2x2 最大池化层
    convolution2dLayer(3, 128, 'Padding', 'same') % 第二个卷积层
    reluLayer
    maxPooling2dLayer(2, 'Stride', 1)
    fullyConnectedLayer(512) % 全连接层，输出 256 个节点
    reluLayer
    fullyConnectedLayer(1) % 输出层 (回归分数)
    regressionLayer % 回归损失层
];

% 设置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 训练模型
fprintf('开始训练模型...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

% 保存训练后的模型
trainedModelPath = 'C:\Program Files\MATLAB\R2024a\code\trainedCNNmodelfor_directt_minus.mat';
save(trainedModelPath, 'net');
fprintf('训练完成，模型已保存到 %s\n', trainedModelPath);

% 模型测试
YPred = predict(net, XTest);

% 计算均方误差 (MSE) 评价模型
mseValue = mean((YPred - YTest).^2);
fprintf('测试集上的均方误差 (MSE): %.4f\n', mseValue);

% 计算线性相关性 (皮尔逊相关系数)
correlationValue = corr(YPred(:), YTest(:)); % 展平以确保兼容
fprintf('测试集上的线性相关性 (皮尔逊相关系数): %.4f\n', correlationValue);

% 输出测试集样本编号和预测结果
resultTable = table(testSampleIDs, YTest, YPred, ...
    'VariableNames', {'SampleID', 'TrueValue', 'PredictedValue'});
disp(resultTable);

% 可视化预测结果
figure;
scatter(YTest, YPred);
xlabel('真实值');
ylabel('预测值');
title('测试集预测结果');
grid on;
