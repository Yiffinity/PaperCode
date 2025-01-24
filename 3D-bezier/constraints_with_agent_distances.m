function [c, ceq] = constraints_with_agent_distances(vars, original_points, obstacle_areas, min_distance, all_curves, agent_idx)
    % vars: 当前智能体的优化变量（控制点）
    % original_points: 当前智能体的原始控制点
    % obstacle_areas: 障碍区域定义
    % min_distance: 智能体之间的最小安全距离
    % all_curves: 所有智能体当前的贝塞尔曲线
    % agent_idx: 当前智能体的索引

    % 初始化约束
    ceq = [];  % 无等式约束

    % 重新构造当前智能体的控制点
    control_points = reshape(vars, [], 3);

    % 生成当前智能体的贝塞尔曲线
    [x, y, z] = generate_bezier(control_points);

    % 获取曲线采样点数
    T = length(x);  % 采样点数量
    M = size(obstacle_areas, 1);  % 障碍区域数量
    N = length(all_curves);  % 智能体数量

    % 初始化固定大小的约束向量
    num_constraints = T * M + T * (N - 1);
    c = zeros(num_constraints, 1);
    constraint_idx = 1;

    % 1. 贝塞尔曲线与障碍区域的最小距离约束
    for i = 1:T  % 遍历采样点
        for j = 1:M  % 遍历障碍区域
            obs_x = obstacle_areas(j, 1);
            obs_y = obstacle_areas(j, 2);
            obs_z = obstacle_areas(j, 3);

            % 点到障碍区域边界的最小距离
            dist = point_to_rect_distance(x(i), y(i), z(i),obs_x, obs_y, obs_z,0.6);
            c(constraint_idx) = min_distance - dist;  % 违反约束为正值
            constraint_idx = constraint_idx + 1;
        end
    end

    % 2. 智能体间的距离约束
    for i = 1:T  % 遍历采样点
        current_position = [x(i), y(i), z(i)];
        for other_agent = 1:N
            if other_agent == agent_idx
                continue; % 跳过当前智能体
            end
            other_position = all_curves{other_agent}(i, :);  % 其他智能体在同一时间步的坐标
            dist = norm(current_position - other_position);
            c(constraint_idx) = min_distance - dist;  % 违反约束为正值
            constraint_idx = constraint_idx + 1;
        end
    end
end
