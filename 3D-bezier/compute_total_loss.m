% 计算优化后的总损失函数
function total_loss = compute_total_loss(original_points, optimized_points, obstacle_areas, min_distance)
    % 1. 偏移损失
    adjust_loss = sum(vecnorm(optimized_points - original_points, 2, 2));

    % % 2. 障碍物惩罚
    % [x, y] = generate_bezier(optimized_points);
    % obstacle_loss = 0;
    % for i = 1:length(x)
    %     for j = 1:size(obstacle_areas, 1)
    %         obs_x = obstacle_areas(j, 1);
    %         obs_y = obstacle_areas(j, 2);
    %         obs_w = obstacle_areas(j, 3);
    %         obs_h = obstacle_areas(j, 4);
    % 
    %         % 点到障碍区域边界的最小距离
    %         dist = point_to_rect_distance(x(i), y(i), obs_x, obs_y, obs_w, obs_h);
    %         if dist < min_distance
    %             obstacle_loss = obstacle_loss + (min_distance - dist)^2;
    %         end
    %     end
    % end

    % 3. 总损失
    % total_loss = adjust_loss + obstacle_loss;
    total_loss = adjust_loss;
end