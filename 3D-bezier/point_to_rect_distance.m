function dist = point_to_rect_distance(x, y, z, rx, ry, rz, re)
    % 计算点 (x, y, z) 到长方体障碍物的最小距离
    % (rx, ry, rz): 长方体的中心坐标
    % re: 长方体的边长（假设为正方体）
    
    % 计算长方体的边界
    xmin = rx - re/2;
    xmax = rx + re/2;
    ymin = ry - re/2;
    ymax = ry + re/2;
    zmin = rz - re/2;
    zmax = rz + re/2;
    
    % 计算 x、y 和 z 方向上的最小距离
    dx = max([xmin - x, 0, x - xmax]);
    dy = max([ymin - y, 0, y - ymax]);
    dz = max([zmin - z, 0, z - zmax]);
    
    % 最小距离为三维空间中的总距离（即欧几里得距离）
    dist = sqrt(dx^2 + dy^2 + dz^2);
end
