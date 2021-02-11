function [u, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x, a, y, rho, p, delta_o, delta_u)
% This function solves the optimal consumption decision when one hides
% Parameters: Cobb-Douglas utility parameter a, expenditure in the revealed
% state (one state lower), price of unobservable good p, cost of hiding 
% delta_o, delta_u, difference in income rho

% Uncomment to run as script
%{
x = 0.6;
a = 0.5;
y = 0.6;
rho = 1;
p = 1.5;
delta_o = 0.5;
delta_u = 0.9;
%}

% When one hides, there are four types of consumption:
% revealed part can be decomposed into observable and unobservable
% consumption
% observable_1 + p*unobservable_1 = pareto weight * income_{state-1}
% Note that consumption must be between 0 and if all budget were spent on
% that good
c_o_1 = a.*y - (1-a).*rho.*delta_o + p.*x.*((1-a).*delta_o./delta_u+a);
c_o_2 = (rho - p.*x./delta_u).*delta_o;

% unrevealed part can be decomposed into observable and unobservable
% consumption
% observable_2/delta_observable + p*unobservable_2/delta_unobservable =
% extra income
c_u_1 = (1-a).*rho.*delta_o./p + (1-a).*y./p - x.*((1-a).*delta_o./delta_u+a);
c_u_2 = x;


% corner solutions: if some consumption is negative, replace with corner
% solution
if c_o_1 < 0
    c_o_1 = 0;
    c_u_1 = y./p;
end

if c_o_2 < 0
    c_o_2 = 0;
    c_u_2 = rho.*delta_u./p;
end

if c_u_1 < 0
    c_u_1 = 0;
    c_o_1 = y;
end

if c_u_2 < 0
    c_u_2 = 0;
    c_o_2 = rho.*delta_o;
end    

% (negative) utility
u = -a.*log(c_o_1+c_o_2) - (1-a).*log(c_u_1+c_u_2);

% for checking: budget constraints
% the revealed consumption should equal income in revealed state (=0)
bc1 = c_o_1 + c_u_1.*p - y;
% the hidden consumption should equal hidden income (=0)
bc2 = c_o_2./delta_o + c_u_2.*p/delta_u - rho;
% total consumption
c_o = c_o_1+c_o_2;
c_u = c_u_1+c_u_2;
% total consumption shuold be less than if no hiding is necessary (<0)
bc3 = c_o + c_u.*p - rho - y;
% utility should be higher than if weren't hiding (>0)
bc4 = -u - log(y) + (1-a).*log(p) - a.*log(a) - (1-a).*log(1-a);
% utility higher than naive strategy of optimizing observable consumption
% and then spending all hidden income on unobservables;
bc5 = -u - a.*log(a.*y) - (1-a).*log((1-a).*y + rho.*delta_u./p);
% effective price for observable and unobservable
p1 = (c_o_1 + c_o_2./delta_o)./(c_o_1 + c_o_2);
p2 = p.*(c_u_1 + c_u_2./delta_u)./(c_u_1 + c_u_2);
% does effective price agree with solution (=0, interior solution)
c_o_test = a.*(y + rho)./p1 - c_o_1 - c_o_2;
c_u_test = (1-a).*(y+rho)./p2 - c_u_1 - c_u_2;

end