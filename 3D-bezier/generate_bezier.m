% 生成三维贝塞尔曲线
function [x, y, z] = generate_bezier(points)
    % 提取三维坐标点
    P = points(:, 1)';
    Q = points(:, 2)';
    R = points(:, 3)';
    
    n = length(P); % 控制点的数量
    t = 0:0.01:1;  % 参数 t 从 0 到 1
    x = zeros(1, length(t));
    y = zeros(1, length(t));
    z = zeros(1, length(t));
    
    for k = 1:length(t)
        % 初始化 p, q, r 为当前控制点
        p = P;
        q = Q;
        r = R;
        
        % 递归计算每个坐标轴上的点
        for j = 2:n
            for i = 1:(n-j+1)
                p(i) = (1-t(k)) * p(i) + t(k) * p(i+1);
                q(i) = (1-t(k)) * q(i) + t(k) * q(i+1);
                r(i) = (1-t(k)) * r(i) + t(k) * r(i+1);
            end
        end
        
        % 存储计算结果
        
        x(k) = p(1);
        y(k) = q(1);
        z(k) = r(1);
    end
end
