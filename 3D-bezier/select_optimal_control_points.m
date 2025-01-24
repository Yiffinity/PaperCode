function selected_indices = select_optimal_control_points(points, num_selected_points, obstacle_areas, min_distance, all_curves, agent_idx)
    n = size(points, 1);
    start_idx = 1;  % 起点固定
    end_idx = n;    % 终点固定
    remaining_indices = 2:(n-1);  % 除去起点和终点的点

    % 枚举所有可能的选择
    combinations = nchoosek(remaining_indices, num_selected_points);

    % 初始化最优选择
    best_loss = inf;
    best_indices = [];

    % 遍历所有组合
    for i = 1:size(combinations, 1)
        temp_indices = [start_idx, combinations(i, :), end_idx];
        temp_points = points(temp_indices, :);

        % 优化当前选择的控制点
        optimized_points = optimize_control_points(temp_points, points, obstacle_areas, min_distance, all_curves, agent_idx);

        % 计算优化后的损失（与原始点的偏移量）
        temp_loss = compute_total_loss(temp_points, optimized_points, obstacle_areas, min_distance);

        % 如果当前组合的损失更小，则更新
        if temp_loss < best_loss
            best_loss = temp_loss;
            best_indices = temp_indices;
        end
    end

    selected_indices = best_indices;
end
