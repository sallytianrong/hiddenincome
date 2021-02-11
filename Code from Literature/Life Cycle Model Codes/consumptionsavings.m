% Simple consumption savings problem, no uncertainty

%% Set up
%Declare parameters
beta = 0.98; r = 0.02; sigma = 0.5; T = 80; rho = 2;

%Define y_t. Note that Y(t) as defined in the matrix is actually y_(t-1).
%This indexing change holds for all of y_t, b_t, and a_t.
y = zeros([1 T+1]);
for t = 1:46
    y(t) = -3 + 0.25*(t-1+20) - 0.0025*(t-1+20)^2;
end

%Define b_t.
B = zeros([1 T+1]);
B(1) = 0;

%utility function
utility = @(c) c.^(1-rho)./(1-rho);

%Savings grid
n = 1000; bgrid = linspace(1e-3,60,n);

%% Value function iteration
%Set up arrays to store value functions, consumption rules, and policy
%functions
V = zeros([length(bgrid),T+1]);
C = zeros([length(bgrid),T+1]);
G = zeros([length(bgrid),T+1]);
%Terminal value function
V(:,T+1) = 0;
%Compute V and C for each time period
for t = T:-1:1
    %Loop over all possible values of a
    for i = 1:length(bgrid)
        % consumption
        c_test = y(t) + bgrid(i) - bgrid'./(1+r);
        penalty = (c_test<=0);
        % Bellman equation 
        X = utility(c_test) + beta.*V(:,t+1) + penalty*(-1e50);
        %Value function is the max over X, policy function is the index
        [V(i,t), G(i,t)] = max(X);
        %Consumption is a function of a
        C(i,t) = y(t) + bgrid(i) - bgrid(G(i,t))./(1+r);
    end
end

%% Find the lifetime wealth path
% I is a path of indices indicating optimal savings behavior.
I = zeros([1 T+1]);
I(1) = 1;
% b_path is lifetime savings path
b_path = zeros([1 T+1]);
% c_path is lifetime consumption path
c_path = zeros([1 T+1]);

for t = 1:T
    I(t+1) = G(I(t),t);
    b_path(t) = bgrid(I(t));
    c_path(t) = C(I(t),t);
end    

% plot
figure;
x = 20:100;
plot(x,y,x,c_path,x,b_path);
legend('Income','Consumption','Savings')

%% Plot consumption function as a function of cash on hand
% define cash on hand
