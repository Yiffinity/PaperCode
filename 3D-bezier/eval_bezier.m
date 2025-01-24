% Function Definitions
function value = eval_bezier(curve_poly, t)
    % Evaluate symbolic Bezier curve (position) at time t
    value(1) = double(subs(curve_poly(1), t)); % x-direction
    value(2) = double(subs(curve_poly(2), t)); % y-direction
    value(3) = double(subs(curve_poly(3), t)); % z-direction
end