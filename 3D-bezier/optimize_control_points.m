% 优化控制点的函数
function optimized_points = optimize_control_points(selected_points, original_points, obstacle_areas, min_distance, all_curves, agent_idx)
   % 优化控制点的函数
    % selected_points: 当前智能体选定的控制点
    % original_points: 当前智能体的原始控制点
    % obstacle_areas: 障碍区域定义
    % min_distance: 智能体之间的最小安全距离
    % all_curves: 所有智能体当前的贝塞尔曲线
    % agent_idx: 当前智能体的索引

    % 提取需要优化的控制点
    initial_points = selected_points;  % 初始控制点

    % 定义目标函数：最小化控制点的偏移量
    objective = @(x) norm(reshape(x, [], 3) - initial_points, 'fro')^2;

    % 定义非线性约束
    nonlcon = @(x) constraints_with_agent_distances(x, original_points, obstacle_areas, min_distance, all_curves, agent_idx);

    % 优化设置
    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');

    % 优化
    optimized_vars = fmincon(objective, initial_points(:), [], [], [], [], [], [], nonlcon, options);

    % 将优化后的控制点重构为矩阵形式
    optimized_points = reshape(optimized_vars, [], 3);
end