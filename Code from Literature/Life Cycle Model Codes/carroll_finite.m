function [C, V, G, d_hat] = carroll_finite(sigma, phi, beta, rho, r, g, T)
% This function solves the finite consumption-savings problem as
% described by Carroll (1997). (mu, sigma) describe the parameters of the
% normal distribution that epsilon follows. Delta is the parameter used to
% calculate beta. rho is the coefficient of risk aversion. r is the
% interest rate. g is the growth rate. e is the error tolerance.

%% Discretize epsilon_t using Tauchen method
% Record parameters. epsilon_t is normal(mu,sigma^2) where theta^2 =
% sigma^2/(1-phi^2).
theta = sigma.^2/(1-phi.^2);
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

%% Create a grid of normalized savings d_hat
dmin = 1e-3; d_star = 1; dmax = 12;
% Create both a fine grid and a coarse grid
fine_grid = linspace(dmin,d_star,100);
coarse_grid = linspace(d_star,dmax,400);
d_hat = [fine_grid'; coarse_grid'];

%% Set up the question
% Parameters
R = 1+r;

% Power utility
u = @(c) c.^(1-rho)./(1-rho);

% Value function
C = zeros(length(d_hat),m-1,T);
G = zeros(length(d_hat),m-1,T);
V = zeros(length(d_hat),m-1,T+1);
V(:,:,T+1) = 0;

%% Value function iteration (finite horizon)

for t = T:-1:1
    for i = 1:length(d_hat) % loop through each gridpoint of d
        for j = 1:m-1 % loop through (m-1) states for epsilon
            c_test = d_hat(i) + 1 - d_hat.*exp(epsilon(j))./R;
            % Penalize negative consumption
            z1 = (c_test<=0);
            % Calculate expectation
            next_value = zeros(length(d_hat),m-1);
            for l = 1:m-1
                next_value(:,l) = exp(epsilon(l)).^(1-rho).*g^(1-rho).*V(:,l,t+1);
            end
            expectation = next_value*Trans(:,1);
            % Maximize
            X = u(c_test) + beta.*expectation + z1.*(-1e50);
            % Value function and policy function
            [V(i,j,t), G(i,j,t)] = max(X);
            % Compute normalized consumption
            C(i,j,t) = d_hat(i) + 1 - d_hat(G(i,j,t)).*exp(epsilon(j))./R;
        end
    end
end