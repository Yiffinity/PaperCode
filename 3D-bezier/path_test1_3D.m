% 初始化每个智能体的轨迹
agent_1_state = [
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

agent_2_state = [
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

agent_3_state = [
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

agent_4_state = [
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

agent_5_state = [
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
                                                        
% 绘制智能体的轨迹
figure;
hold on;
plot3(agent_1_state(:, 1), agent_1_state(:, 2), agent_1_state(:, 3), '-o', 'DisplayName', 'Agent 1', 'LineWidth', 2);
plot3(agent_2_state(:, 1), agent_2_state(:, 2), agent_2_state(:, 3), '-s', 'DisplayName', 'Agent 2', 'LineWidth', 2);
plot3(agent_3_state(:, 1), agent_3_state(:, 2), agent_3_state(:, 3), '-^', 'DisplayName', 'Agent 3', 'LineWidth', 2);
plot3(agent_4_state(:, 1), agent_4_state(:, 2), agent_4_state(:, 3), '-d', 'DisplayName', 'Agent 4', 'LineWidth', 2);
plot3(agent_5_state(:, 1), agent_5_state(:, 2), agent_5_state(:, 3), '-x', 'DisplayName', 'Agent 5', 'LineWidth', 2);

view(3);
% 设置图形
xlabel('X');
ylabel('Y');
zlabel('Z');
legend;
grid on;

