% Incentive compatible allocations under one sided hiding
% key elements: 1) consumption allocation 2) future expected utility

%% Parameters
% set seed
rng(4);

% Parameters
% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% Pareto weight
alpha = 0.5;

% Time discounting
beta = 0.96;

% Cost of hiding
delta = 0.8;

% agent 2's income
y2 = 3;

%% Functional forms
% Income process is uniform on the interval [ymin,ymax]. agent 2's income
% is fixed.
ymin = 1; ymax = 5;

% to simplify, let's say there are only two states of income, with equal
% probability
n = 50;
income = [ymin ymax];

% agent 2 chooses how much to transfer to agent 1 (can be negative)
% the most they can transfer is all of agent 2's income. the least they can
% transfer is taking all of agent 1's income.
transfer = linspace(-ymax,y2,n);

%% Grid of promised utility values
% grid of promised utility values is bounded below by autarky value and
% above by the agent 1 consume the entire endowment every period
Vmin = 0.5.*(log(ymin) + log(ymax))./(1-beta);
Vmax = 0.5.*(log(ymin+y2) + log(ymax+y2))./(1-beta);
V = linspace(Vmin,Vmax,n);

% Lower bound of U: constant transfers
cons_trans = zeros(1,n);
for i = 1:n
   cond = @(x) log(income(1)+x)./(1-beta) + log(income(2)+x)./(1-beta) - 2.*V(i);
   cons_trans(i) = fzero(cond,3);
end
% not feasible to always transfer more than y2
feasible_lb = (cons_trans<y2);
U_lb = log(y2-cons_trans(feasible_lb))./(1-beta);

% Upper bound of U: perfect risk-sharing
cons_c = exp((1-beta).*V);
% not feasible to always guarantee consumption more than ymin+y2
feasible_ub = (cons_c<ymin+y2);
U_ub = 0.5.*(log(y2 - cons_c(feasible_ub) + ymin) + log(y2 - cons_c(feasible_ub) + ymax))./(1-beta);

%% First find the admissable space of transfers T and continuation utility W, given each V gridpoint

% save number of possible transfers
counter_num = zeros(1,n);

% for each V gridpoint
for i = 1:n
    
    counter = 0;
    % Find all possible transfer schemes
    % high income state
    for t2 = 1:n
        % in the low income state, transfer must be at least as high as the
        % high income state
        for t1 = t2:n
            
            % Transfers
            T_test(1) = max(transfer(t1),-ymin);
            T_test(2) = transfer(t2);
            
            % Calculate W given V and T
            W_test = zeros(1,2);
            V_test = zeros(1,n);
            
            % loop through all gridpoints to find starting value of W
            for j = 1:n
                
                % starting value of W
                W_test(1) = V(j);
                
                % calculate W
                W_test(2) = W_test(1) + log(income(2) + T_test(1))./beta - log(income(2) + T_test(2))./beta;
                
                % calculate promised utility
                V_test(j) = mean(log(income+T_test) + beta.*W_test);
            end
            
            % is this feasible to choose W so that we arrive at V? (promise
            % keeping constraint)
            feasible = (min(abs(V_test - V(i)))<1);
            counter = counter + feasible;
            
            % if feasible
            if feasible == 1
                
                % find the appropriate starting point of W
                [~, index] = min(abs(V_test - V(i)));
                
                % calculate W and T
                T(counter, 1, i) = T_test(1);
                T(counter, 2, i) = T_test(2);
                W(counter, 1, i) = V(index);
                W(counter, 2, i) = W(counter, 1, i) + log(income(2) + T_test(1))./beta - log(income(2) + T_test(2))./beta;
                W_index(counter, 1, i) = index;
                [~, W_index(counter, 2, i)] = min(abs(W(counter,2)-V));
            end
        end
    end
    counter_num(i) = counter;
end

%% Make some T and W schedules
%T_schedule = permute(T(1,:,:),[3 2 1]);
%W_schedule = permute(W_index(1,:,:),[3 2 1]);

%% Taking V as given, agent 2 maximizes utility
%{
Paut = y2./(1-beta);
P_test = ones(1,n).*Paut;
diff = 1;

while diff>0.001
    P_update = (log(y2-T_schedule(:,1)) + beta.*P_test(W_schedule(:,1)) + log(y2-T_schedule(:,2)) + beta.*P_test(W_schedule(:,2)))./2;
    diff = max(abs(P_update-P_test));
    P_test = P_update;
end
%}