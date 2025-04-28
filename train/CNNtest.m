% 清空环境
clear; clc;

% 1. 加载测试数据
load("C:\Program Files\MATLAB\R2024a\code\CNN\data_all\ten\mpccd.mat");  % 读取特征数据
load("C:\Program Files\MATLAB\R2024a\code\CNN\data_all\ten\MPCCD_MOS.mat");  % 读取分数数据

%% icip
%if istable(icip_tensor)
%    XTest = table2array(icip_tensor);  % 将表格转换为数组
%else
%    XTest = icip_tensor;  % 如果已经是数组，直接使用
%end
%
%if istable(SJTU_MOS)
%    YTest = table2array(SJTU_MOS);  % 将表格转换为数组
%else
%    YTest = SJTU_MOS;  % 如果已经是数组，直接使用
%end
%% SJTU
%if istable(mpccd_tensor)
%    XTest = table2array(mpccd_tensor);  % 将表格转换为数组
%else
%    XTest = mpccd_tensor;  % 如果已经是数组，直接使用
%end
%
%if istable(SJTU_MOS)
%    YTest = table2array(SJTU_MOS);  % 将表格转换为数组
%else
%    YTest = SJTU_MOS;  % 如果已经是数组，直接使用
%end
%% SJTU
%if istable(sjtu_tensor)
%    XTest = table2array(sjtu_tensor);  % 将表格转换为数组
%else
%    XTest = sjtu_tensor;  % 如果已经是数组，直接使用
%end
%
%if istable(SJTU_MOS)
%    YTest = table2array(SJTU_MOS);  % 将表格转换为数组
%else
%    YTest = SJTU_MOS;  % 如果已经是数组，直接使用
%end
%% SJTU
%if istable(wpc_tensor)
%    XTest = table2array(wpc_tensor);  % 将表格转换为数组
%else
%    XTest = wpc_tensor;  % 如果已经是数组，直接使用
%end
%
%if istable(SJTU_MOS)
%    YTest = table2array(SJTU_MOS);  % 将表格转换为数组
%else
%    YTest = SJTU_MOS;  % 如果已经是数组，直接使用
%end
%%
XTest = mpccd_tensor;
YTest = MPCCD_MOS;

% 2. 加载保存的模型
modelFilePath = 'C:\Program Files\MATLAB\R2024a\code\CNN\trainedCNNModel.mat'; % 模型保存路径
loadedData = load(modelFilePath); % 加载模型文件
net = loadedData.net; % 提取保存的网络对象

% 3. 确保数据形状一致
% 需要确保 XTest 的维度是 [5, 3, 3, 90]，即 [高度, 宽度, 通道数, 样本数]
XTest = permute(XTest, [2, 3, 4, 1]); % 将 XTest 重排为 [5, 3, 3, 90]

% 4. 使用模型进行预测
YPred = predict(net, XTest);

% 5. 计算均方误差 (MSE)
mseValue = mean((YPred - YTest).^2);
fprintf('测试数据上的均方误差 (MSE): %.4f\n', mseValue);

% 6. 计算皮尔逊相关系数
correlationValue = corr(YPred(:), YTest(:)); % 展平以确保兼容
fprintf('测试数据上的线性相关性 (皮尔逊相关系数): %.4f\n', correlationValue);

% 7. 计算 Spearman's Rank Correlation Coefficient (SRCC)
srccValue = corr(YTest(:), YPred(:), 'Type', 'Spearman');
fprintf('测试数据上的秩相关性 (SRCC): %.4f\n', srccValue);

% 9. 计算 Kendall's Tau
kendallValue = corr(YTest(:), YPred(:), 'Type', 'Kendall');
fprintf('测试数据上的 Kendall Tau: %.4f\n', kendallValue);

% 7. 可视化预测结果
figure;
scatter(YTest, YPred);
xlabel('真实值');
ylabel('预测值');
title('测试数据预测结果');
grid on;
