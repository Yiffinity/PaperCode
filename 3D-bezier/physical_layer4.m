%增加拜占庭点攻击2.0
% Parameters
num_agents = length(all_curves_poly); % Number of agents based on workspace variable
gamma = 0.1 * (1:num_agents); % Adaptation rate
Gamma = 0.5 * (1:num_agents); % Adaptation rate
m = 1 ./ (1:num_agents); 
theta = 0.1 * (1:num_agents);

% Initial Conditions
x = zeros(num_agents, 6); % [x_i, x_dot_i, y_i, y_dot_i, z_i, z_dot_i] for all agents
x(:,1) = 1; % Random initial x positions
x(:,3) = 1; % Random initial y positions
x(:,5) = 1; % Random initial z positions
theta_hat = 2 * (1:num_agents); % Initial estimates of theta
b_hat = ones(1, num_agents); % Initial estimates of b
b_hat_x =0.2 * (1:num_agents); % Initial estimates of b
b_hat_y = 0.2 * (1:num_agents); % Initial estimates of b
b_hat_z =0.2 * (1:num_agents); % Initial estimates of b
% k1 = 3 * ones(1, num_agents); % Initial gain for k1
% k2 = 3 * ones(1, num_agents); % Initial gain for k2
% k1_dot = zeros(1, num_agents); % Derivative of k1
% k2_dot = zeros(1, num_agents); % Derivative of k2

k1_x = 3 * ones(1, num_agents); % Initial gain for k1
k2_x = 3 * ones(1, num_agents); % Initial gain for k2
k1_dot_x = zeros(1, num_agents); % Derivative of k1
k2_dot_x = zeros(1, num_agents); % Derivative of k2

k1_y = 3 * ones(1, num_agents); % Initial gain for k1
k2_y = 3 * ones(1, num_agents); % Initial gain for k2
k1_dot_y = zeros(1, num_agents); % Derivative of k1
k2_dot_y = zeros(1, num_agents); % Derivative of k2

k1_z = 3 * ones(1, num_agents); % Initial gain for k1
k2_z = 3 * ones(1, num_agents); % Initial gain for k2
k1_dot_z = zeros(1, num_agents); % Derivative of k1
k2_dot_z = zeros(1, num_agents); % Derivative of k2

rho_hat_x = zeros(1, num_agents); % Estimated Byzantine attack amplitude
rho_hat_y = zeros(1, num_agents); % Estimated Byzantine attack amplitude
rho_hat_z = zeros(1, num_agents); % Estimated Byzantine attack amplitude
time_span = 0:0.00005:1; % Simulation time

rho_hat = zeros(1, num_agents); % Estimated Byzantine attack amplitude
omega = 1e-2; % Small constant for softsign
bar_d = 0.1; % Threshold for attack amplitude estimation
A = [0 1; 0 0]; % State matrix for 3D space
B = [0; 1]; % Input matrix for 3D space
Q = [1 0; 0 1]; % State weighting matrix
R = 1; % Input weighting scalar

% Compute P for each agent and direction (including z direction)
P_x = zeros(2, 2, num_agents); % Riccati matrices for x direction
P_y = zeros(2, 2, num_agents); % Riccati matrices for y direction
P_z = zeros(2, 2, num_agents); % Riccati matrices for z direction

% P = eye(3); % Riccati matrix, precomputed
for i = 1:num_agents
    % Solve Riccati equation for x, y, and z directions
    B = [0; i];
    P_x(:,:,i) = care(A, B, Q, R);
    P_y(:,:,i) = care(A, B, Q, R);
    P_z(:,:,i) = care(A, B, Q, R);
end

% Attacked agent (simulate Byzantine attack)
attacked_agent = 1; % Specify which agent is under Byzantine attack
attack_magnitude = 1; % Magnitude of the Byzantine attack

% Desired trajectory functions using all_curves_poly
s = @(t, i) eval_bezier(all_curves_poly{i}, t); % Desired trajectory
s_dot = @(t, i) eval_bezier_derivative(all_curves_poly{i}, t, 1); % First derivative
s_ddot = @(t, i) eval_bezier_derivative(all_curves_poly{i}, t, 2); % Second derivative

% Dynamics
f = @(x2, i) -abs(x2) .* x2; % Nonlinearity

% Simulation
x_hist = zeros(num_agents, 6, length(time_span)); % History of states
u_hist = zeros(num_agents, 3, length(time_span)); % Control input history for x, y, z
theta_hat_hist = zeros(num_agents, length(time_span)); % Estimated theta history
theta_hat_hist_x = zeros(num_agents, length(time_span)); % Estimated theta history for x
theta_hat_hist_y = zeros(num_agents, length(time_span)); % Estimated theta history for y
theta_hat_hist_z = zeros(num_agents, length(time_span)); % Estimated theta history for z

% Simulation
k1_hist_x = zeros(num_agents, length(time_span)); % History of k1
k2_hist_x = zeros(num_agents, length(time_span)); % History of k2
k1_hist_y = zeros(num_agents, length(time_span)); % History of k1
k2_hist_y = zeros(num_agents, length(time_span)); % History of k2
k1_hist_z = zeros(num_agents, length(time_span)); % History of k1
k2_hist_z = zeros(num_agents, length(time_span)); % History of k2

for t_idx = 1:length(time_span)
    t = time_span(t_idx);
    
    for i = 1:num_agents

        B = [0; i];

        % Desired Trajectory for Agent i
        s_t = s(t, i);
        s_t_dot = s_dot(t, i);
        s_t_ddot = s_ddot(t, i);

        % Errors (x and y directions)
        delta1_x = x(i,1) - s_t(1);
        delta2_x = x(i,2) - (s_t_dot(1) - k1_x(i) * delta1_x);

        delta1_y = x(i,3) - s_t(2);
        delta2_y = x(i,4) - (s_t_dot(2) - k1_y(i) * delta1_y);

        delta1_z = x(i,5) - s_t(3);
        delta2_z = x(i,6) - (s_t_dot(3) - k1_z(i) * delta1_z);

        if i==4
            % Adaptive gains update laws
            k1_dot_x(i) = gamma(i) * (abs(delta1_x)) * (1 + 0.1 * (abs(delta1_x)));
            k2_dot_x(i) = gamma(i) * (abs(delta2_x)) * (1 + 0.1 * (abs(delta2_x)));
            k1_x(i) = max(1, min(k1_x(i) + k1_dot_x(i) * 0.005, 10)); % Lower and upper bounds for k1
            k2_x(i) = max(1, min(k2_x(i) + k2_dot_x(i) * 0.005, 10)); % Lower and upper bounds for k2
        else
            % Adaptive gains update laws
            k1_dot_x(i) = gamma(i) * (abs(delta1_x)) * (1 + 0.5 * (abs(delta1_x)));
            k2_dot_x(i) = gamma(i) * (abs(delta2_x)) * (1 + 0.5 * (abs(delta2_x)));
            k1_x(i) = max(1, min(k1_x(i) + k1_dot_x(i) * 0.05, 30)); % Lower and upper bounds for k1
            k2_x(i) = max(1, min(k2_x(i) + k2_dot_x(i) * 0.05, 30)); % Lower and upper bounds for k2
        end
        
        % Adaptive gains update laws
        k1_dot_y(i) = gamma(i) * (abs(delta1_y)) * (1 + 0.2 * (abs(delta1_y)));
        k2_dot_y(i) = gamma(i) * (abs(delta2_y)) * (1 + 0.2 * (abs(delta2_y)));
        k1_y(i) = max(1, min(k1_y(i) + k1_dot_y(i) * 0.02, 30)); % Lower and upper bounds for k1
        k2_y(i) = max(1, min(k2_y(i) + k2_dot_y(i) * 0.02, 30)); % Lower and upper bounds for k2

        % Adaptive gains update laws
        k1_dot_z(i) = gamma(i) * (abs(delta1_z)) * (1 + 0.2 * (abs(delta1_z)));
        k2_dot_z(i) = gamma(i) * (abs(delta2_z)) * (1 + 0.2 * (abs(delta2_z)));
        k1_z(i) = max(1, min(k1_z(i) + k1_dot_z(i) * 0.02, 30)); % Lower and upper bounds for k1
        k2_z(i) = max(1, min(k2_z(i) + k2_dot_z(i) * 0.02, 30)); % Lower and upper bounds for k2

        % Auxiliary Control Variables (x and y directions)
        pi1_x = s_t_dot(1) - k1_x(i) * delta1_x;
        pi2_x = -delta1_x - k2_x(i) * delta2_x + s_t_ddot(1) - k1_x(i) * delta2_x + k1_x(i)^2 * delta1_x;

        pi1_y = s_t_dot(2) - k1_y(i) * delta1_y;
        pi2_y = -delta1_y - k2_y(i) * delta2_y + s_t_ddot(2) - k1_y(i) * delta2_y + k1_y(i)^2 * delta1_y;

        pi1_z = s_t_dot(3) - k1_z(i) * delta1_z;
        pi2_z = -delta1_z - k2_z(i) * delta2_z + s_t_ddot(3) - k1_z(i) * delta2_z + k1_z(i)^2 * delta1_z;

        % Attack Compensation Calculation
        sigma_x = [x(i,1) - s_t(1); x(i,2) - s_t_dot(1)]; % Error state vector for x
        sigma_y = [x(i,3) - s_t(2); x(i,4) - s_t_dot(2)]; % Error state vector for y
        sigma_z = [x(i,5) - s_t(3); x(i,6) - s_t_dot(3)]; % Error state vector for z
        
         % Norm and softsign for attack compensation
        P_x_i = P_x(:,:,i); % Riccati matrix for x
        P_y_i = P_y(:,:,i); % Riccati matrix for y
        P_z_i = P_z(:,:,i); % Riccati matrix for z
        chi_hat_x = (sigma_x' * P_x_i * B) / (norm(sigma_x' * P_x_i * B) + omega) * rho_hat_x(i);
        chi_hat_y = (sigma_y' * P_y_i * B) / (norm(sigma_y' * P_y_i * B) + omega) * rho_hat_y(i);
        chi_hat_z = (sigma_z' * P_z_i * B) / (norm(sigma_z' * P_z_i * B) + omega) * rho_hat_z(i);

         % Update rho_hat (attack estimation)
        sigma_p_b_norm_x = norm(sigma_x' * P_x_i * B);
        sigma_p_b_norm_y = norm(sigma_y' * P_y_i * B);
        sigma_p_b_norm_z = norm(sigma_z' * P_z_i * B);

        if sigma_p_b_norm_x >= bar_d
            rho_hat_dot_x = sigma_p_b_norm_x + 2 * omega;
        else
            rho_hat_dot_x = sigma_p_b_norm_x + 2 * omega * (sigma_p_b_norm_x / bar_d);
        end

        if sigma_p_b_norm_y >= bar_d
            rho_hat_dot_y = sigma_p_b_norm_y + 2 * omega;
        else
            rho_hat_dot_y = sigma_p_b_norm_y + 2 * omega * (sigma_p_b_norm_y / bar_d);
        end

        if sigma_p_b_norm_z >= bar_d
            rho_hat_dot_z = sigma_p_b_norm_z + 2 * omega;
        else
            rho_hat_dot_z = sigma_p_b_norm_z + 2 * omega * (sigma_p_b_norm_z / bar_d);
        end

        rho_hat_x(i) = rho_hat_x(i) + rho_hat_dot_x * 0.01; % Euler integration
        rho_hat_y(i) = rho_hat_y(i) + rho_hat_dot_y * 0.01; % Euler integration
        rho_hat_z(i) = rho_hat_z(i) + rho_hat_dot_z * 0.01; % Euler integration
        % Update b_hat and theta_hat
        b_hat_dot_x = -gamma(i) * (pi2_x * delta2_x); % Combined x and y contributions
        theta_hat_dot_x = Gamma(i) * (1/m(i)) * (f(x(i,2), i) * delta2_x);

        b_hat_dot_y = -gamma(i) * (pi2_y * delta2_y); % Combined x and y contributions
        theta_hat_dot_y = Gamma(i) * (1/m(i)) * (f(x(i,4), i) * delta2_y);

        b_hat_dot_z = -gamma(i) * (pi2_z * delta2_z); % Combined x and y contributions
        theta_hat_dot_z = Gamma(i) * (1/m(i)) * (f(x(i,6), i) * delta2_z);

       
        % b_hat(i) = b_hat(i) + b_hat_dot * 0.01; % Euler integration
        % theta_hat(i) = theta_hat(i) + theta_hat_dot * 0.01; % Euler integration

        % Control Inputs for x and y directions
        % u_x = b_hat_x(i) * pi2_x - chi_hat_x;
        % u_y = b_hat_y(i) * pi2_y - chi_hat_y;
        % u_z = b_hat_z(i) * pi2_z - chi_hat_z;
        u_x = b_hat_x(i) * pi2_x;
        u_y = b_hat_y(i) * pi2_y;
        u_z = b_hat_z(i) * pi2_z;

        % Simulate Byzantine attack for a specific agent
        % if i == attacked_agent
        %     u_x = u_x + attack_magnitude * sin(2*pi*t); % Add sinusoidal attack
        %     u_y = u_y + attack_magnitude * sin(2*pi*t); % Add sinusoidal attack
        %     u_z = u_z + attack_magnitude * sin(2*pi*t); % Add sinusoidal attack
        % end


        u_hist(i, :, t_idx) = [u_x, u_y, u_z];

        % System Dynamics Update (x and y directions)
        x_dot1 = x(i,2);
        x_dot2 = (1/m(i)) * u_x + (1/m(i)) * f(x(i,2), i) * theta(i);

        y_dot1 = x(i,4);
        y_dot2 = (1/m(i)) * u_y + (1/m(i)) * f(x(i,4), i) * theta(i);

        z_dot1 = x(i,6);
        z_dot2 = (1/m(i)) * u_z + (1/m(i)) * f(x(i,6), i) * theta(i);

        x(i,1) = x(i,1) + x_dot1 * 0.01; % Update x position
        x(i,2) = x(i,2) + x_dot2 * 0.01; % Update x velocity
        x(i,3) = x(i,3) + y_dot1 * 0.01; % Update y position
        x(i,4) = x(i,4) + y_dot2 * 0.01; % Update y velocity
        x(i,5) = x(i,5) + z_dot1 * 0.01; % Update z position
        x(i,6) = x(i,6) + z_dot2 * 0.01; % Update z velocity
    end
    x_hist(:,:,t_idx) = x; % Save state history
    k1_hist_x(:,t_idx) = k1_x; % Save k1 history
    k2_hist_x(:,t_idx) = k2_x; % Save k2 history
    k1_hist_y(:,t_idx) = k1_y; % Save k1 history
    k2_hist_y(:,t_idx) = k2_y; % Save k2 history
    k1_hist_z(:,t_idx) = k1_z; % Save k1 history
    k2_hist_z(:,t_idx) = k2_z; % Save k2 history
end

% % Plot Results for x and y Directions
% figure();
% for i = 1:num_agents
%     subplot(num_agents, 1, i);
%     hold on;
%     % Actual trajectories
%     plot(squeeze(x_hist(i,1,:)), squeeze(x_hist(i,3,:)), 'r', 'DisplayName', 'Actual Trajectory','LineWidth', 1.5);
%     % Desired trajectories
%     desired_trajectory = arrayfun(@(t) s(t, i), time_span, 'UniformOutput', false);
%     desired_x = cellfun(@(v) v(1), desired_trajectory); % x-direction
%     desired_y = cellfun(@(v) v(2), desired_trajectory); % y-direction
%     plot(desired_x, desired_y, 'b--', 'DisplayName', 'Desired Trajectory','LineWidth', 1.5);
%     title(['Agent ' num2str(i) ' Trajectory Tracking']);
%     xlabel('x');
%     ylabel('y');
%     xlim([0, 9]); % Restrict x-axis range to [1, 8]
%     ylim([0, 9]); % Restrict y-axis range to [1, 8]
%     legend;
% end
% % 障碍区定义
%     obstacle_areas = [
%         2.5, 2.5, 1, 3;
%         5.5, 2.5, 1, 3;
%         3.5, 4.5, 2, 1;
%     ];
% figure;
% % Plot Results for x and y Directions
% for agent_idx = 1:num_agents
%     % 绘制实际轨迹
%     plot(squeeze(x_hist(agent_idx,1,:)), squeeze(x_hist(agent_idx,3,:)), ...
%         'r', 'DisplayName', 'Actual Trajectory', 'LineWidth', 1.5);
%     hold on;
% 
%     % 绘制目标轨迹
%     desired_trajectory = arrayfun(@(t) s(t, agent_idx), time_span, 'UniformOutput', false);
%     desired_x = cellfun(@(v) v(1), desired_trajectory); % x-direction
%     desired_y = cellfun(@(v) v(2), desired_trajectory); % y-direction
%     plot(desired_x, desired_y, 'b--', 'DisplayName', 'Desired Trajectory', 'LineWidth', 1.5);
%     hold on;
% 
%     % 设置图形属性
%     title(['Agent ' num2str(agent_idx) ' Trajectory Tracking']);
%     xlabel('x');
%     ylabel('y');
%     legend;
%     xlim([0, 9]); % 限制 x 轴范围
%     ylim([0, 9]); % 限制 y 轴范围
%     grid on; % 打开网格线
% end
% % 绘制障碍区域
% for obs_idx = 1:size(obstacle_areas, 1)
%     rectangle('Position', obstacle_areas(obs_idx, :), ...
%         'EdgeColor', 'r', 'LineWidth', 0.1, 'LineStyle', '--', ...
%         'FaceColor', [1 0 0 0.3]);
% end
% % 绘制附加蓝色矩形
% rectangle('Position', [0.5, 0.5, 1, 1], ...
%     'EdgeColor', 'b', 'LineWidth', 0.1, 'LineStyle', '--', ...
%     'FaceColor', [0 0 1 0.3]);
% rectangle('Position', [7.5, 7.5, 1, 1], ...
%     'EdgeColor', 'b', 'LineWidth', 0.1, 'LineStyle', '--', ...
%     'FaceColor', [0 0 1 0.3]);
% 
% Initialize error storage
error_x = zeros(num_agents, length(time_span)); % x-direction error
error_y = zeros(num_agents, length(time_span)); % y-direction error
error_z = zeros(num_agents, length(time_span)); % y-direction error

% Calculate errors over time
for t_idx = 1:length(time_span)
    t = time_span(t_idx);
    for i = 1:num_agents
        % Desired trajectory at time t
        s_t = s(t, i); 

        % Calculate errors in x and y directions
        error_x(i, t_idx) = (x_hist(i,1,t_idx) - s_t(1)); % Error in x-direction
        error_y(i, t_idx) = (x_hist(i,3,t_idx) - s_t(2)); % Error in y-direction
        error_z(i, t_idx) = (x_hist(i,5,t_idx) - s_t(3)); % Error in y-direction
    end
end
figure;
num_colors = 6; % 计算需要的颜色数量
colors = lines(num_colors); % 生成基本颜色
% colors = {'r', 'g', 'b', 'y'};
% Plot errors for each agent
for i = 1:5
    % Plot x-direction error
    subplot(3, 1, 1);
    plot(time_span, error_x(i, :), 'LineWidth', 2,'Color',colors(i, :)); % 设置线宽并选择颜色
    hold on;
    title('Error in x-direction', 'FontSize', 18, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    xlabel('Time (s)', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    ylabel('Error (x)', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    grid on;
    ylim([-0.5 1])
    % xlim([0 0.55])
    xlim([0 0.5])
    set(gca, 'FontSize', 14, 'LineWidth', 1.3);

    % Plot y-direction error
    subplot(3, 1, 2);
    plot(time_span, error_y(i, :), 'LineWidth', 2,'Color',colors(i,:)); % 设置线宽并选择颜色
    hold on;
    title('Error in y-direction', 'FontSize', 18, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    xlabel('Time (s)', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    ylabel('Error (y)', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    grid on;
    xlim([0 0.5])
    set(gca, 'FontSize', 14, 'LineWidth', 1.3);

    % Plot z-direction error
    subplot(3, 1, 3);
    plot(time_span, error_z(i, :), 'LineWidth', 2, 'Color', colors(i, :)); % 设置线宽并选择颜色
    hold on;
    title('Error in z-direction', 'FontSize', 18, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    xlabel('Time (s)', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    ylabel('Error (z)', 'FontSize', 16, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    grid on;
    ylim([-0.5 1])
    xlim([0 0.5])
    % xlim([0 0.55])
    set(gca, 'FontSize', 19, 'LineWidth', 1.3);
end

% Adjust figure properties
set(gcf, 'Units', 'centimeters', 'Position', [1, 1, 20*1, 20*0.8]); % 调整图形大小
set(gca, 'LooseInset', [0, 0, 0.04, 0]); % 调整坐标轴的边距

% figure;
% % Plot errors for each agent
% for i = 1:1
%     % Plot x-direction error
%     subplot(2, 1, 1);
%     plot(time_span, error_x(i, :), 'LineWidth', 1.5);
%     hold on
%     title(' Error in x-direction');
%     xlabel('Time (s)');
%     ylabel('Error (x)');
%     grid on;
% 
%     % Plot y-direction error
%     subplot(2, 1, 2);
%     plot(time_span, error_y(i, :), 'LineWidth', 1.5);
%     hold on
%     title(' Error in y-direction');
%     xlabel('Time (s)');
%     ylabel('Error (y)');
%     grid on;
% end
figure
for i = 1:5
    % 绘制每个智能体的误差曲线
    plot(time_span, error_x(i, :), 'LineWidth', 2, 'Color', colors(i, :)); % 设置线宽并选择颜色
    hold on;
    
    % 设置图形标签
    xlabel('Time (s)', 'FontSize', 18, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    ylabel('Tracking Error along X-Dimension', 'FontSize', 18, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    
    % 设置网格和轴范围
    grid on;
    ylim([-0.5 1]);
    xlim([0 0.5]);
    
    % 设置坐标轴字体和线宽
    set(gca, 'FontSize', 14, 'LineWidth', 1.3);

    % 设置 x 轴数字显示为当前值×10
    xticks = get(gca, 'xtick'); % 获取当前的 x 轴刻度
    xticklabels = arrayfun(@(x) sprintf('%.1f', x*10), xticks, 'UniformOutput', false); % 将 x 轴的数字乘以 10
    set(gca, 'xticklabel', xticklabels); % 更新 x 轴的刻度标签

end

% 添加图例，分别为 UAV1 到 UAV5
legend({'UAV 1', 'UAV 2', 'UAV 3', 'UAV 4', 'UAV 5'}, 'FontSize', 14, 'FontName', 'Times New Roman');
