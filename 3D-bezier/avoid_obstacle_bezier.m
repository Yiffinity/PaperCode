global agent_num;
agent_num = 5;
% 将五个智能体的state数据放入一个名为state_marks的cell数组中
state_marks = cell(5, 1);

state_marks{1} = [
    [0 0 0];
    [1 0 1];
    [2 0 2];
    [3 0 3];
    [4 0 4];
    [5 0 5];
    [6 1 5];
    [7 1 6];
    [7 2 6];
    [7 3 6];
    [7 4 6];
    [7 5 6];
    [7 6 6];
    [7 7 6];
    [7 7 6];
    [7 7 7];
    [7 7 7]
];

state_marks{2} = [
    [0 0 0];
    [1 1 1];
    [2 2 2];
    [3 3 3];
    [4 4 4];
    [5 5 4];
    [6 6 4];
    [7 7 4];
    [7 7 5];
    [7 7 6];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7]
];

state_marks{3} = [
    [0 0 0];
    [1 1 1];
    [2 2 2];
    [3 3 3];
    [4 4 4];
    [5 5 5];
    [6 6 6];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7]
];

state_marks{4} = [
    [0 0 0];
    [1 0 1];
    [2 0 2];
    [3 0 3];
    [4 0 4];
    [5 0 5];
    [6 0 6];
    [7 1 6];
    [7 2 7];
    [7 3 7];
    [7 4 7];
    [7 5 7];
    [7 6 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7]
];

state_marks{5} = [
    [0 0 0];
    [1 1 1];
    [2 2 2];
    [3 3 3];
    [4 4 4];
    [5 5 5];
    [6 6 6];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7];
    [7 7 7]
];

% 定义智能体轨迹
agent_trajectories = state_marks;
% 为每个智能体分配颜色
agent_colors = lines(agent_num);  % 使用 MATLAB 的 colormap 'lines' 生成不同颜色

% 障碍物坐标
obstacle_areas = [
    2 1 2;
    2 1 3;
    2 2 3;
    2 3 2;
    2 3 3;
];

min_distance = 0.3;  % 设置最小距离
num_selected_points = 3; % 除起点和终点外选择的控制点数量

% 初始化所有智能体的贝塞尔曲线
all_curves = cell(agent_num, 1);
% 初始化所有智能体的贝塞尔曲线多项式
all_curves_poly = cell(agent_num, 1);

% 初始化曲线为原始点的线性插值
for idx = 1:agent_num
    points = agent_trajectories{idx};
    [x, y, z] = generate_bezier(points);  % 使用原始点生成初始曲线
    all_curves{idx} = [x', y', z'];
end

for idx = 1:agent_num
    points = agent_trajectories{idx};

    % % 优化选择初始控制点
    % selected_indices = select_optimal_control_points(points, num_selected_points, obstacle_areas, min_distance);
    % 优化选择初始控制点
    selected_indices = select_optimal_control_points(points, num_selected_points, obstacle_areas, min_distance, all_curves, idx);


    % 提取选定的控制点
    selected_points = points(selected_indices, :);

    % 使用优化算法调整控制点
    % optimized_points = optimize_control_points(selected_points, points, obstacle_areas, min_distance);
    % 使用优化算法调整控制点
    optimized_points = optimize_control_points(selected_points, points, obstacle_areas, min_distance, all_curves, idx);

    % 生成贝塞尔曲线
    [x, y, z] = generate_bezier(optimized_points);

    [x_poly, y_poly,z_poly] = bezier_polynomial(optimized_points);

    % 保存贝塞尔曲线
    all_curves{idx} = [x', y', z'];
    % 保存贝塞尔曲线多项式
    all_curves_poly{idx} = [x_poly',y_poly',z_poly'];

    % % 绘制轨迹
    % plot(points(:, 1), points(:, 2), 'b--o', 'MarkerSize', 10, 'MarkerFaceColor', 'w');
    % hold on;
    % plot(x, y, 'y', 'LineWidth', 2);
    % 绘制原始轨迹
    if idx == 1
        plot3(points(:, 1), points(:, 2), points(:, 3), '--', 'LineWidth', 1, 'Color', agent_colors(idx, :), ...
            'DisplayName', sprintf('Agent''s Original Path'));
    else
        plot3(points(:, 1), points(:, 2), points(:, 3), '--', 'LineWidth', 1, 'Color', agent_colors(idx, :));
    end
    hold on;

    if idx == 1
        % 绘制优化后的轨迹
        plot3(x, y, z, '-', 'LineWidth', 2, 'Color', agent_colors(idx, :), ...
            'DisplayName', sprintf('Agent''s Optimized Path'));
    else
        % 绘制优化后的轨迹
        plot3(x, y, z, '-', 'LineWidth', 2, 'Color', agent_colors(idx, :));
    end

    if idx == 1
        % 绘制选定的控制点
        scatter3(selected_points(:, 1), selected_points(:, 2), selected_points(:, 3), 50, agent_colors(idx, :), 'o', 'filled', ...
            'DisplayName', sprintf('Agent''s Selected Points'));
    else
        % 绘制选定的控制点
        scatter3(selected_points(:, 1), selected_points(:, 2), selected_points(:, 3), 50, agent_colors(idx, :), 'o', 'filled');
    end

    if idx == 1
        % 绘制优化的控制点
        scatter3(optimized_points(:, 1), optimized_points(:, 2), optimized_points(:, 3), 70, agent_colors(idx, :), '^', 'filled', ...
            'DisplayName', sprintf('Agent''s Optimized Points'));
    else
        % 绘制优化的控制点
        scatter3(optimized_points(:, 1), optimized_points(:, 2), optimized_points(:, 3), 70, agent_colors(idx, :), '^', 'filled');
    end

end
% 添加图例
legend('show', 'Location', 'best','FontSize', 18);

hold on
% 绘制障碍物（灰色正方体）
for i = 1:size(obstacle_areas, 1)
    % 获取当前障碍物的中心点
    center = obstacle_areas(i, :);
    
    % 定义立方体的顶点（单位长度，中心点为`center`）
%     x = [0 1 1 0 0 1 1 0] - 0.5; % X轴
%     y = [0 0 1 1 0 0 1 1] - 0.5; % Y轴
%     z = [0 0 0 0 1 1 1 1] - 0.5; % Z轴
    x = [0 1 1 0 0 1 1 0]*0.6 - 0.3; % X轴
    y = [0 0 1 1 0 0 1 1]*0.6 - 0.3; % Y轴
    z = [0 0 0 0 1 1 1 1]*0.6 - 0.3; % Z轴
    
    % 平移立方体到障碍物的中心
    x = x + center(1);
    y = y + center(2);
    z = z + center(3);
    
    % 使用 patch 绘制每个立方体的六个面
    faces = [
        1 2 6 5; 1 2 3 4; 1 4 8 5; 5 6 7 8; 3 4 8 7; 2 3 7 6
    ];
    patch('Vertices', [x' y' z'], 'Faces', faces, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');
end

% 设置图形属性
grid on;
shg;
set(gcf, 'Units', 'centimeters', 'Position', [1, 1, 20*1, 20*0.8]);
set(gca, 'FontSize', 18);
set(gca, 'LineWidth', 1.3);

% 设置视角
view(3); % 设置为3D视角

% 绘制多项式 与 求导
% plot_and_differentiate(all_curves_poly);