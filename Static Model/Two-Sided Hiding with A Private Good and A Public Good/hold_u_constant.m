function [c_self, c_other, Q] = hold_u_constant(agent,u,y,a1,a2,price)
% This function inputs one agent's utility and budget constraints. It
% outputs the optimal allocation to maximize the other agent's utility.
% agent = 1 or 2 is the agent whose utility u is to be held constant

% set up
if agent==1
    a_self = a1;
    a_other = a2;
else
    a_self = a2;
    a_other = a1;
end

% fmincon options
options = optimoptions('fmincon','Display','off');

% Lower bound: public good consumption must be nonnegative
z_lb = exp(u./a_self - (1-a_self).*log(y)./a_self);

% Nonlinear inequality constraint: the other agent's consumption must be
% nonnegative
function [c,ceq] = constraint(z)
    c = - y + z + price.*exp((u-a_self.*log(z))./(1-a_self));
    ceq = [];
end

% Find an appropriate starting value for optimization
z_init = linspace(z_lb, y, 10);
c_init = y - z_init - price.*exp((u-a_self.*log(z_init))./(1-a_self));
z_init = z_init(c_init>=0);
z_final_init = z_init(round(length(z_init)/2,0));

% Maximize other agent's utility
u_other = @(z) -a_other.*log(y - z - price.*exp((u-a_self.*log(z))./(1-a_self))) - (1-a_other).*(u-a_self.*log(z))./(1-a_self);
[c_self, ~] = fmincon(u_other,z_final_init,[],[],[],[],z_lb,y,@constraint,options);

% Calculate allocation
c_other = y - c_self - price.*exp((u-a_self.*log(c_self))./(1-a_self));
Q = exp((u-a_self.*log(c_self))./(1-a_self));

end