function [x_poly, y_poly, z_poly] = bezier_polynomial_3d(control_points)
    % 输入:
    % control_points: n x 3 的控制点矩阵，每行是 (x, y, z)
    % 输出:
    % x_poly: 贝塞尔曲线 x 方向的多项式表达式
    % y_poly: 贝塞尔曲线 y 方向的多项式表达式
    % z_poly: 贝塞尔曲线 z 方向的多项式表达式

    n = size(control_points, 1) - 1;  % 贝塞尔曲线的阶数

    % 提取 x, y 和 z 控制点
    x_control = control_points(:, 1)';
    y_control = control_points(:, 2)';
    z_control = control_points(:, 3)';

    % 初始化符号变量
    syms t;
    x_poly = 0;
    y_poly = 0;
    z_poly = 0;

    % 计算贝塞尔曲线的多项式
    for i = 0:n
        % 组合数 C(n, i)
        coeff = nchoosek(n, i);

        % 贝塞尔基函数
        basis = coeff * (1 - t)^(n - i) * t^i;

        % 累加到多项式中
        x_poly = x_poly + x_control(i + 1) * basis;
        y_poly = y_poly + y_control(i + 1) * basis;
        z_poly = z_poly + z_control(i + 1) * basis;
    end

    % 简化多项式
    x_poly = simplify(x_poly);
    y_poly = simplify(y_poly);
    z_poly = simplify(z_poly);

    % 显示结果
    fprintf('贝塞尔曲线的多项式表达式:\n');
    fprintf('x(t) = %s\n', char(x_poly));
    fprintf('y(t) = %s\n', char(y_poly));
    fprintf('z(t) = %s\n', char(z_poly));
end
