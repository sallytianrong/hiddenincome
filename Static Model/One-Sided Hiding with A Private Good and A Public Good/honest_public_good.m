function [eu, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_public_good(x,a1,a2,price,delta,alpha,y,rho,p)

% fmincon options: turn on/off display
%options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
options = optimoptions('fmincon','Display','off');
% Multistart setup
rng default % For reproducibility
ms = MultiStart;
ms.Display = 'off';

% Matrices to store allocation
c_hc = zeros(3,1);
d_hc = zeros(3,1);
Q_hc = zeros(3,1);
u_hc = zeros(3,1);
v_hc = zeros(3,1);

% In the first state, there are two variables: x1 = private good for agent
% 1, x2 = private good for agent 2
% public good expenditure equals the rest of the budget
c_hc(1) = x(1);
d_hc(1) = x(2);
Q_hc(1) = (y(1) - x(1) - x(2))./price;
% utility
u_hc(1) = a1.*log(c_hc(1))+(1-a1).*log(Q_hc(1));
v_hc(1) = a2.*log(d_hc(1))+(1-a2).*log(Q_hc(1));

% In the second state, calculate agent 1's hiding utility. The honest
% utility must be at least as high.
u_hc(2) = a1.*log(c_hc(1)+delta*rho)+(1-a1).*log(Q_hc(1));

% Find an appropriate starting value for optimization
z_init_1 = linspace(c_hc(1),c_hc(1)+delta*rho,10);
z_lb_1 = exp(u_hc(2)./a1 - (1-a1).*log(y(2))./a1);
z_init_1 = z_init_1(z_init_1>=z_lb_1);
d_init_1 = y(2) - z_init_1 - price.*exp((u_hc(2)-a1.*log(z_init_1))./(1-a1));
z_init_1 = z_init_1(d_init_1>=0);
z_final_init1 = z_init_1(1);

% Nonlinear inequality constraints
function [c,ceq] = constraint1(z)
    c = - y(2) + z + price.*exp((u_hc(2)-a1.*log(z))./(1-a1));
    ceq = [];
end

% Maximize agent 2's utility, holding agent 1's utility constant
v_2 = @(z) -a2.*log(y(2) - z - price.*exp((u_hc(2)-a1.*log(z))./(1-a1))) - (1-a2).*(u_hc(2)-a1.*log(z))./(1-a1);
%problem = createOptimProblem('fmincon','x0',z_final_init1,'objective',v_2,'lb',0,'ub',y(2),'nonlcon',@constraint1);
%[c_hc(2),v_2] = run(ms,problem,3);
[c_hc(2), v_2] = fmincon(v_2,z_final_init1,[],[],[],[],0,y(2),@constraint1,options);
v_hc(2) = -v_2;
% Calculate allocation
d_hc(2) = y(2) - c_hc(2) - price.*exp((u_hc(2)-a1.*log(c_hc(2)))./(1-a1));
Q_hc(2) = exp((u_hc(2)-a1.*log(c_hc(2)))./(1-a1));

% In the third state, calculate agent 1's hiding utility. The honest
% utility must be at least as high.
u_hc(3) = a1.*log(c_hc(2)+delta*rho)+(1-a1).*log(Q_hc(2));

% Find an appropriate starting value
z_init_2 = linspace(c_hc(2),c_hc(2)+delta*rho,10);
z_lb_2 = exp(u_hc(3)./a1 - (1-a1).*log(y(2))./a1);
z_init_2 = z_init_2(z_init_2>=z_lb_2);
d_init_2 = y(3) - z_init_2 - price.*exp((u_hc(3)-a1.*log(z_init_2))./(1-a1));
z_init_2 = z_init_2(d_init_2>=0);
z_final_init2 = z_init_2(1);

% Nonlinear inequality constraints
function [c,ceq] = constraint2(z)
    c = - y(3) + z + price.*exp((u_hc(3)-a1.*log(z))./(1-a1));
    ceq = [];
end

% Maximize agent 2's utility, holding agent 1's utility constant
v_3 = @(z) -a2.*log(y(3) - z - price.*exp((u_hc(3)-a1.*log(z))./(1-a1))) - (1-a2).*(u_hc(3)-a1.*log(z))./(1-a1);
%problem = createOptimProblem('fmincon','x0',z_final_init2,'objective',v_3,'lb',0,'ub',y(3),'nonlcon',@constraint2);
%[c_hc(3), v_3] = run(ms,problem,3);
[c_hc(3), v_3] = fmincon(v_3,z_final_init2,[],[],[],[],0,y(3),@constraint2,options);
v_hc(3) = -v_3;
% Calculate allocation
d_hc(3) = y(3) - c_hc(3) - price.*exp((u_hc(3)-a1.*log(c_hc(3)))./(1-a1));
Q_hc(3) = exp((u_hc(3)-a1.*log(c_hc(3)))./(1-a1));

% Calculate utility
h_hc = alpha.*u_hc + (1-alpha).*v_hc;
% expected utility
eu = - p*h_hc;

end
