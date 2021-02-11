% Replicate Pistaferri (2019) MPC graphs

clear all; close all; clc;
%% Parameters
% interest rate
r = 0.04;
% discount factor
beta = 0.95;
% relative risk aversion
rho = 2;
% standard deviation of the permanent shock
phi = 0.95;
sigma = 0.01;
% time
T = 60;
% growth rate of income
g = 1;

%% Calculate solutions
[C, V, G, d_hat] = carroll_finite(sigma, phi, beta, rho, r, g, T);

%% Simulate T periods of income, consumption and savings

% Discretize epsilon_t using Tauchen method
% Record parameters. epsilon_t is normal(mu,sigma^2) where theta^2 =
% sigma^2/(1-phi^2).

% Pick (m-1) gridpoints such that the successive areas under the standard
%nomrmal are equal to 1/m.
m = 9; a = zeros([m+1 1]);
a(1) = -Inf; a(m+1) = Inf;
for i = 2:m
    a(i) = norminv((i-1)/m);
end
% Calculate the transition matrix following Deaton(1991)
Trans = zeros(m-1);
for i = 1:m-1
    for j = 1:m-1
        f = @(x) exp(-x.^2./(2.*theta.^2)).*(normcdf((theta.*a(j+1)-phi.*x)./sigma)-normcdf((theta.*a(j)-phi.*x)./sigma));
        int = integral(f,theta*a(i),theta*a(i+1));
        Trans(i,j) = (1/(theta*sqrt(2*pi)))*int;
    end
end
% Scale all the rows such that they each sum to one
sumT = sum(Trans,2);
Trans = Trans./sumT;
% Income shock in each state
epsilon = theta.*a(2:m);

% Simulate Markov Chain for epsilon (permanent income shocks)
n = 60; s0 = 4;
sim=rand(n,1); state_e = zeros(n,1); state_e(1)= s0;
cumsum_tran=[zeros(m-1,1) cumsum(Trans')'];
for t=2:n
    state_e(t)=find(((sim(t)<=cumsum_tran(state_e(t-1),2:m))&(sim(t)>cumsum_tran(state_e(t-1),1:m-1))),1);
end
epsilon_chain = epsilon(state_e);

% Find simulated income path
Perm = zeros(n,1); g = 1;
Perm(1) = 20000;
for t = 2:n
    Perm(t) = g.*exp(epsilon_chain(t)).*Perm(t-1);
end
Income = Perm;

% Find cash on hand and consumption paths
d_path = zeros(n,1); c_path = zeros(n,1);
% Initial savings is zero. Since we don't have zero in our grid, we pick a numerical approximate to zero. 
d_path(1) = 1e-3;
for t = 2:n
    % Find the gridpoint that reflects last period's savings
    [~, index_d] = min(abs(d_hat - d_path(t-1)));
    % Find next period's savings using the policy function
    d_path(t) = d_hat(G(index_d,state_e(t-1),t));
    c_path(t-1) = d_path(t-1) + 1 - d_path(t).*epsilon_chain(t-1)./(1+r);
end

% Find total consumption and savings
consumption_path = c_path.*Perm;
savings_path = d_path.*Perm;

%% Plot simulated results
figure; 
plot(1:n, Perm(1:n), 1:n, Income(1:n));
title('Simulated Income');
legend('Permanent Income', 'Realized Income')

figure;
plot(1:n, d_path(1:n), 1:n, c_path(1:n))
title('Simulated Normalized Consumption and Savings')
legend('Normalized Savings','Normalized Consumption')

figure;
plot(1:n, Income(1:n), 1:n, savings_path(1:n), 1:n, consumption_path(1:n))
title('Simulated Income, Consumption and Savings')
legend('Income', 'Savings','Consumption')

%% Calculate MPC
% set age = 40
C_40 = C(:,:,40);
% MPC = 