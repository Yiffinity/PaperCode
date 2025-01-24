clc;
clear;
close all;

s=[ 1 2; 1 3; 1 4; 2 4;2 5;2 6; 2 7;3 7; 4 7; 5 7; 6 7;6 8;7 8;8 8];

% 定义控制点，确保曲线穿越障碍区
controlPoints = [1 2; 1 3; 1 4; 2 4; 8 8];

% 定义控制点，确保曲线穿越障碍区
controlPoints1 = [1 2; 2 7; 3 7; 7 8; 8 8];

% 绘制控制点
hold on;
% plot(controlPoints(:,1), controlPoints(:,2), 'bo-', 'MarkerFaceColor','b'); 
% 绘制贝塞尔曲线
t = linspace(0, 1, 100); % 生成100个样本点
bezierCurve = bezier(controlPoints, t); % 计算贝塞尔曲线

% 绘制贝塞尔曲线，使用蓝色
plot(bezierCurve(:,1), bezierCurve(:,2), 'Color', [191, 29, 45]/256, 'LineWidth', 2); 

hold on;

% 计算贝塞尔曲线
bezierCurve1 = bezier(controlPoints1, t); 

% 绘制贝塞尔曲线，使用黄色
plot(bezierCurve1(:,1), bezierCurve1(:,2), 'Color', [79, 189, 129]/256, 'LineWidth', 2); 


% 添加障碍物
obstacle = [3 3; 3 6; 6 6; 6 3]; % 障碍物的四个角
fill(obstacle(:,1), obstacle(:,2), [214, 214, 214]/256, 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % 绘制障碍物


% 绘制障碍物的边界（加粗红色虚线）
hold on;
plot([obstacle(:,1); obstacle(1,1)], [obstacle(:,2); obstacle(1,2)], '--','Color',[214,214,214]/256, 'LineWidth', 2);
hold on;
% 绘制状态点
plot(s(:,1), s(:,2), 'o', 'MarkerFaceColor','k','MarkerSize', 5); % 状态点

% 用虚线连接状态点
hold on;
plot(s(:,1), s(:,2), 'Color','k', 'LineStyle', '--', 'LineWidth', 1.5); % 用浅蓝色虚线连接
hold on;
plot(controlPoints(:,1), controlPoints(:,2), '^','Color', [191, 29, 45]/256, 'MarkerFaceColor',[191, 29, 45]/256, 'MarkerSize', 8);
hold on;
plot(controlPoints1(:,1), controlPoints1(:,2), '^','Color', [79, 189, 129]/256, 'MarkerFaceColor', [79, 189, 129]/256, 'MarkerSize', 8);
hold on;
% 标记START和GOAL
text(1, 1.2, 'START', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold');
text(8.2, 7.8, 'GOAL', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 16, 'FontWeight', 'bold');
hold on;
% 定义两个点
points = [1 2; 8 8];

% 绘制这些点，使用红色的五角星标注
plot(points(:,1), points(:,2), 'rp', 'MarkerFaceColor','r', 'MarkerSize',12); 

% 设置坐标轴
axis([0.5 8.5 1.5 8.5]);
% % 绘制黑色边框
% rectangle('Position', [0.5, 1.5, 8, 7], 'EdgeColor', 'k', 'LineWidth', 0.5); % 黑色矩形边框

grid on; % 显示网格
axis off; % 关闭坐标轴显示


function bezierCurve = bezier(controlPoints, t)
    % 计算贝塞尔曲线
    n = size(controlPoints, 1) - 1;
    bezierCurve = zeros(length(t), 2);
    for i = 1:length(t)
        B = zeros(1, 2);
        for j = 0:n
            % 贝塞尔基函数
            B = B + nchoosek(n, j) * (1 - t(i))^(n-j) * t(i)^j * controlPoints(j+1,:);
        end
        bezierCurve(i, :) = B;
    end
end
