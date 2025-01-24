function value = eval_bezier_derivative(curve_poly, t, order)
    % Evaluate symbolic Bezier curve derivative at time t
    % curve_poly: 1x2 sym (Bezier curve for x and y directions)
    % t: time variable
    % order: derivative order
    syms tau; % Define symbolic variable for differentiation
    value(1) = double(subs(diff(curve_poly(1), tau, order), tau, t)); % x-direction
    value(2) = double(subs(diff(curve_poly(2), tau, order), tau, t)); % y-direction
    value(3) = double(subs(diff(curve_poly(3), tau, order), tau, t)); % y-direction
end